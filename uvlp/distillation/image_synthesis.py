"""
Image synthesis utilities for UVLP.

This module provides functions for synthesizing images from prototypes
using the Stable Diffusion Latents2Img pipeline.
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

import torch
from torchvision.utils import save_image


def synthesize_images(
    prototypes: List[Dict],
    pipeline,
    config,
    device: str = 'cuda',
) -> torch.Tensor:
    """Synthesize images from prototypes using Latents2Img pipeline.
    
    Args:
        prototypes: List of prototype dictionaries.
        pipeline: Loaded Latents2Img pipeline.
        config: Configuration object.
        device: Device to run synthesis on.
        
    Returns:
        Tensor of synthesized latents.
    """
    pipe_device = next(pipeline.unet.parameters()).device
    pipe_dtype = next(pipeline.unet.parameters()).dtype
    
    gen = torch.Generator(device=pipe_device)
    
    latents_list = []
    for proto in prototypes:
        Zc = proto['image_prototype'].unsqueeze(0).to(device=pipe_device, dtype=pipe_dtype)
        Tc = proto['text_prototype']
        
        out = pipeline(
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
        
        Z_out = out.images if isinstance(out.images, torch.Tensor) else torch.stack(out.images)
        latents_list.append(Z_out.squeeze(0))
    
    return torch.stack(latents_list)


def save_images_for_evaluation(
    latents: torch.Tensor,
    class_wnid: str,
    phase: str,
    decoder,
    out_root: str,
    output_size: int = 256,
) -> str:
    """Decode and save latents as images for evaluation.
    
    Args:
        latents: VAE latents to decode.
        class_wnid: Class WNID.
        phase: Phase name (e.g., 'jvl_c').
        decoder: Object with decode_latents method.
        out_root: Output root directory.
        output_size: Size to resize output images.
        
    Returns:
        Path to the phase output directory.
    """
    save_dir = os.path.join(out_root, f'distilled_images_{phase}', class_wnid)
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.inference_mode():
        images = decoder.decode_latents(latents)
        
        # Optionally resize
        if output_size and images.shape[-1] != output_size:
            import torch.nn.functional as F
            images = F.interpolate(
                images, 
                size=(output_size, output_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        for idx, img in enumerate(images):
            save_image(img, os.path.join(save_dir, f'{idx}.png'))
    
    return os.path.join(out_root, f'distilled_images_{phase}')


def create_image_grid(
    images: torch.Tensor,
    nrow: int = 5,
    padding: int = 2,
) -> torch.Tensor:
    """Create a grid of images for visualization.
    
    Args:
        images: Tensor of images [N, C, H, W].
        nrow: Number of images per row.
        padding: Padding between images.
        
    Returns:
        Grid image tensor.
    """
    from torchvision.utils import make_grid
    return make_grid(images, nrow=nrow, padding=padding, normalize=True)

