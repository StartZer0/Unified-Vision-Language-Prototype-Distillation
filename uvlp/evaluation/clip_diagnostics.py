"""
CLIP-based diagnostics for evaluating distilled images.

This module provides functions for computing CLIP-based metrics
to evaluate the quality of distilled images.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from transformers import CLIPModel, CLIPProcessor
import PIL.Image as Image


def _load_clip_for_diag(device: str) -> Tuple[CLIPModel, CLIPProcessor]:
    """Load CLIP model/processor on the requested device for diagnostics.
    
    Args:
        device: Device to load model on.
        
    Returns:
        Tuple of (CLIP model, CLIP processor).
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    return model.to(device).eval(), proc


@torch.no_grad()
def _gather_class_features(
    root: str,
    wnid: str,
    model: CLIPModel,
    proc: CLIPProcessor,
    device: str,
    max_per_class: int = 64
) -> Optional[torch.Tensor]:
    """Collect up to max_per_class image features from a class folder using CLIP.
    
    Args:
        root: Root directory containing class folders.
        wnid: Class WNID.
        model: CLIP model.
        proc: CLIP processor.
        device: Device to run on.
        max_per_class: Maximum images to process per class.
        
    Returns:
        Tensor of normalized image features, or None if no images found.
    """
    cls_dir = os.path.join(root, wnid)
    if not os.path.isdir(cls_dir):
        return None
    
    imgs = []
    for fn in os.listdir(cls_dir):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                imgs.append(Image.open(os.path.join(cls_dir, fn)).convert('RGB'))
            except Exception:
                continue
            if len(imgs) >= max_per_class:
                break
    
    if len(imgs) == 0:
        return None
    
    inputs = proc(images=imgs, return_tensors='pt', padding=True).to(device)
    feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def _class_text_feature(
    name: str,
    model: CLIPModel,
    proc: CLIPProcessor,
    device: str
) -> torch.Tensor:
    """Compute the normalized CLIP text embedding for a simple class prompt.
    
    Args:
        name: Class name.
        model: CLIP model.
        proc: CLIP processor.
        device: Device to run on.
        
    Returns:
        Normalized text feature tensor.
    """
    prompt = f"a photo of a {name}"
    t = proc(text=[prompt], return_tensors='pt', padding=True, truncation=True).to(device)
    tf = model.get_text_features(**t)
    return tf / tf.norm(dim=-1, keepdim=True)


def compute_clip_diagnostics_for_root(
    syn_root: str,
    class_ids: List[str],
    class_mapping: Dict[str, str],
    device: str = 'cuda',
) -> Dict[str, Dict[str, float]]:
    """Compute CLIP-based diagnostics for distilled images.
    
    Args:
        syn_root: Root directory containing distilled images.
        class_ids: List of class WNIDs.
        class_mapping: Mapping from WNID to class name.
        device: Device to run on.
        
    Returns:
        Dictionary mapping WNID to metrics dict.
    """
    model, proc = _load_clip_for_diag(device)
    
    results = {}
    for wnid in class_ids:
        feats = _gather_class_features(syn_root, wnid, model, proc, device)
        if feats is None:
            continue
        
        name = class_mapping.get(wnid, wnid)
        text_feat = _class_text_feature(name, model, proc, device)
        
        # Intra-class cohesion (average pairwise similarity)
        if feats.size(0) > 1:
            sim_matrix = feats @ feats.t()
            n = feats.size(0)
            intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
        else:
            intra_sim = 1.0
        
        # Text alignment (average similarity to class text)
        text_align = (feats @ text_feat.t()).mean()
        
        results[wnid] = {
            'intra_cohesion': float(intra_sim),
            'text_alignment': float(text_align),
            'num_images': feats.size(0),
        }
    
    return results

