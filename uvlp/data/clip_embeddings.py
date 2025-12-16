"""
CLIP embedding extraction utilities.

This module provides functions for extracting and processing CLIP image
and text embeddings for the UVLP framework.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel


def extract_clip_embeddings(
    images: torch.Tensor,
    texts: List[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized CLIP image and text embeddings for a batch of samples.
    
    Args:
        images: Batch of images as tensors [B, C, H, W].
        texts: List of text descriptions.
        clip_model: Pre-loaded CLIP model.
        clip_processor: Pre-loaded CLIP processor.
        device: Device to run computations on.
        
    Returns:
        Tuple of (normalized image features, normalized text features).
    """
    to_pil = transforms.ToPILImage()
    
    with torch.no_grad():
        image_inputs = clip_processor(
            images=[to_pil(img) for img in images],
            return_tensors='pt',
            padding=True,
        ).to(device)

        text_inputs = clip_processor(
            text=texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(device)

        image_features = clip_model.get_image_features(**image_inputs)
        text_features = clip_model.get_text_features(**text_inputs)
        
        # L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        
    return image_features, text_features


@torch.no_grad()
def build_prompt_ensemble_name_tfs(
    class_ids: List[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    class_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, torch.Tensor]:
    """Construct prompt-ensembled text prototypes for each class.

    Uses multiple prompt templates to create robust class text embeddings.

    Args:
        class_ids: List of class WNIDs.
        clip_model: Pre-loaded CLIP model.
        clip_processor: Pre-loaded CLIP processor.
        device: Device to run computations on.
        class_mapping: Mapping from WNID to human-readable class name.
            If None, uses default IMAGEWOOF_CLASS_MAPPING.

    Returns:
        Dictionary mapping WNID to normalized text feature tensor.
    """
    from uvlp.data.dataset_loader import IMAGEWOOF_CLASS_MAPPING

    if class_mapping is None:
        class_mapping = IMAGEWOOF_CLASS_MAPPING

    templates = [
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a black and white photo of a {}.',
        'a low contrast photo of a {}.',
        'a high contrast photo of a {}.',
        'a bad photo of a {}.',
        'a good photo of a {}.',
        'a photo of a small {}.',
        'a photo of a big {}.',
        'a photo of the {}.',
        'a blurry photo of the {}.',
        'a black and white photo of the {}.',
        'a low contrast photo of the {}.',
        'a high contrast photo of the {}.',
        'a bad photo of the {}.',
        'a good photo of the {}.',
        'a photo of the small {}.',
        'a photo of the big {}.',
    ]
    
    name_tf: Dict[str, torch.Tensor] = {}
    
    for wnid in class_ids:
        name = class_mapping.get(wnid, wnid)
        texts = [tmpl.format(name) for tmpl in templates]
        
        text_inputs = clip_processor(
            text=texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        feats = clip_model.get_text_features(**text_inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        
        # Average across templates
        mean_feat = feats.mean(dim=0)
        mean_feat = mean_feat / mean_feat.norm().clamp_min(1e-8)
        
        name_tf[wnid] = mean_feat.detach().to(device=device)
    
    return name_tf

