"""
Main distillation pipeline for UVLP.

This module implements the full JVL-C distillation loop that processes
each class and generates synthetic distilled images.
"""

from __future__ import annotations

import os
import math
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import save_image

from uvlp.configs.base_config import Config
from uvlp.clustering.jvl_clustering import JVLClustering
from uvlp.data.dataset_loader import (
    create_imagewoof_dataloader,
    load_text_descriptions,
    get_transform_protocol,
)
from uvlp.data.clip_embeddings import build_prompt_ensemble_name_tfs
from uvlp.optimization.fisher_alpha import refine_alpha_per_class
from uvlp.utils.memory import free_memory
from uvlp.distillation.latents2img import load_latents2img_pipeline


def ensure_class_file(path: str, class_mapping: Optional[Dict[str, str]] = None):
    """Ensure the class file exists, creating it if necessary."""
    from uvlp.data.dataset_loader import IMAGEWOOF_CLASS_MAPPING
    
    if class_mapping is None:
        class_mapping = IMAGEWOOF_CLASS_MAPPING
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        wnids = list(class_mapping.keys())
        with open(path, 'w') as f:
            f.write('\n'.join(wnids) + '\n')
        print(f'Wrote class file: {path}')
    else:
        print(f'Using class file: {path}')


def save_images_for_evaluation(
    latents: torch.Tensor,
    class_wnid: str,
    phase: str,
    clustering: JVLClustering,
    out_root: str
) -> str:
    """Decode and persist a batch of latents for a class/phase.
    
    Args:
        latents: VAE latents to decode.
        class_wnid: Class WNID.
        phase: Phase name (e.g., 'jvl_c').
        clustering: JVLClustering instance for decoding.
        out_root: Output root directory.
        
    Returns:
        Path to the phase output directory.
    """
    save_dir = os.path.join(out_root, f'distilled_images_{phase}', class_wnid)
    os.makedirs(save_dir, exist_ok=True)
    with torch.inference_mode():
        images = clustering.decode_latents(latents)
        for idx, img in enumerate(images):
            save_image(img, os.path.join(save_dir, f'{idx}.png'))
    return os.path.join(out_root, f'distilled_images_{phase}')


def run_jvl_distillation(
    config: Config,
    baseline_repo: str,
    finetuned_model_path: Optional[str] = None
) -> str:
    """Full JVL-C distillation loop across classes.
    
    Args:
        config: Configuration object.
        baseline_repo: Path to the baseline VLCP repository.
        finetuned_model_path: Optional path to fine-tuned model.
        
    Returns:
        Path to the JVL-C output directory.
    """
    torch.manual_seed(config.current_seed)
    np.random.seed(config.current_seed)

    ensure_class_file(config.class_file)

    # Load dataset and text descriptions
    transform = get_transform_protocol(config.resolution)
    dataloader, path_all, dataset_classes = create_imagewoof_dataloader(
        config.data_root, transform, batch_size=10
    )
    descriptions_map = load_text_descriptions(config.metadata_file)

    # Group by WNID
    class_ids = [line.strip() for line in open(config.class_file, 'r') if line.strip()]
    class_data = {wnid: {'images': [], 'labels': [], 'paths': []} for wnid in class_ids}
    
    for images, labels, indices in tqdm(dataloader, desc='Collecting data'):
        paths = [path_all[i] for i in indices]
        for img, label, path in zip(images, labels, paths):
            wnid = dataset_classes[label.item()]
            class_data[wnid]['images'].append(img)
            class_data[wnid]['labels'].append(label)
            class_data[wnid]['paths'].append(path)

    # Initialize clustering utilities
    jvlc = JVLClustering(
        alpha=config.alpha,
        device=config.device,
        diag_checks=True,
        random_state=config.current_seed,
        top_k_visual=config.top_k_visual,
    )

    prompt_name_tfs = build_prompt_ensemble_name_tfs(
        class_ids, jvlc.clip_model, jvlc.clip_processor, config.device,
        class_mapping=None  # Will use default
    )
    tau_neg_cache: Dict[str, Optional[torch.Tensor]] = {}
    for wnid in class_ids:
        negs = [prompt_name_tfs[other] for other in class_ids if other != wnid and other in prompt_name_tfs]
        tau_neg_cache[wnid] = torch.stack(negs, dim=0) if negs else None

    model_path = finetuned_model_path or config.finetuned_model_path
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Fine-tuned model not found at {model_path}.\n"
            "Please run the fine-tuning step to create it."
        )
    l2i_pipe = load_latents2img_pipeline(model_path=model_path)

    print("Latents2Img device:",
          next(l2i_pipe.unet.parameters()).device,
          "dtype:", next(l2i_pipe.unet.parameters()).dtype)

    # Output roots
    os.makedirs(config.distilled_root, exist_ok=True)
    jvl_c_root = os.path.join(config.distilled_root, 'distilled_images_jvl_c')
    os.makedirs(jvl_c_root, exist_ok=True)

    # Process each class
    for wnid in class_ids:
        imgs = class_data[wnid]['images']
        if len(imgs) == 0:
            print(f'No images for class {wnid}')
            continue

        class_tensor = torch.stack(imgs)
        texts = []
        for p in class_data[wnid]['paths']:
            rel = os.path.join(wnid, os.path.basename(p))
            alt = rel.replace(f'_{wnid}.JPEG', '.JPEG')
            text = descriptions_map.get(rel) or descriptions_map.get(alt) or ''
            texts.append(text)

        # Extract features
        vae_latents = jvlc.extract_vae_latents(class_tensor)
        clip_img, clip_txt = jvlc.extract_clip_embeddings(class_tensor, texts)

        tau_pos = prompt_name_tfs[wnid]
        tau_neg_stack = tau_neg_cache.get(wnid)
        r_c = torch.mean(torch.sum(clip_img * clip_txt, dim=-1)).item()
        g = 1.0 / (1.0 + math.exp(-(r_c - 0.25) / 0.05))
        align_scale = 0.3 + 0.7 * g
        a_lo = 0.45 if r_c < 0.25 else config.jvlc_global_alpha_min
        a_hi = config.jvlc_global_alpha_max
        a_lo = min(a_hi, max(config.jvlc_global_alpha_min, a_lo))

        alpha_class = refine_alpha_per_class(
            wnid=wnid,
            img_feats=clip_img,
            txt_feats=clip_txt,
            name_tf=tau_pos,
            global_alpha=config.alpha,
            device=config.device,
            delta=0.1,
        )

        # Generate prototypes via JVL-C
        prototypes, inliers, _ = jvlc.derive_prototypes_jvl_c(
            vae_latents,
            clip_img,
            clip_txt,
            texts,
            ipc=config.ipc,
            contamination=config.contamination,
            alpha_override=alpha_class,
            tau_pos=tau_pos,
            tau_neg_stack=tau_neg_stack,
            align_weight=align_scale,
            wnid=wnid,
        )

        if len(prototypes) == 0:
            print(f'No prototypes for class {wnid}')
            continue

        # Ensure pipeline on correct device and dtype
        try:
            l2i_pipe.to(
                torch_device=config.device,
                torch_dtype=(torch.float16 if config.device == 'cuda' else torch.float32)
            )
        except TypeError:
            l2i_pipe.to(config.device)

        pipe_device = next(l2i_pipe.unet.parameters()).device
        pipe_dtype = next(l2i_pipe.unet.parameters()).dtype

        try:
            l2i_pipe.scheduler.set_timesteps(50, device=pipe_device)
        except TypeError:
            l2i_pipe.scheduler.set_timesteps(50)

        gen = torch.Generator(device=pipe_device)

        # Generate synthesized latents via Latents2Img
        initial_latents_list = []
        for idx, proto in enumerate(prototypes, 1):
            Zc = proto['image_prototype'].unsqueeze(0).to(device=pipe_device, dtype=pipe_dtype)
            Tc = proto['text_prototype']

            out = l2i_pipe(
                prompt=Tc,
                latents=Zc,
                is_init=True,
                strength=config.strength,
                guidance_scale=config.guidance_scale,
                num_inference_steps=50,
                negative_prompt=config.negative_prompt,
                output_type='latent',
                generator=gen,
            )

            Z_init = out.images if isinstance(out.images, torch.Tensor) else torch.stack(out.images)
            initial_latents_list.append(Z_init.squeeze(0))

        initial_latents = torch.stack(initial_latents_list)

        # Save JVL-C images
        save_images_for_evaluation(initial_latents, wnid, 'jvl_c', jvlc, config.distilled_root)

        # Cleanup per class
        objs = [
            class_tensor, vae_latents, clip_img, clip_txt,
            prototypes, initial_latents_list, initial_latents, texts
        ]
        free_memory(*objs)

    # Final cleanup
    free_memory(
        l2i_pipe,
        jvlc,
        dataloader, path_all, descriptions_map, class_data
    )

    print("Distillation complete")
    print(f"  JVL-C images: {jvl_c_root}")
    return jvl_c_root

