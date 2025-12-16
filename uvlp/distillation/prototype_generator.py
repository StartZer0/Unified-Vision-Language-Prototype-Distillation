"""
Prototype generation utilities for UVLP.

This module provides functions for generating prototypes from clustered data.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch


def generate_prototypes(
    vae_latents: torch.Tensor,
    clip_img_embeds: torch.Tensor,
    clip_txt_embeds: torch.Tensor,
    descriptions: List[str],
    labels: np.ndarray,
    ipc: int,
    top_k_visual: int = 6,
    device: str = 'cuda',
) -> List[Dict]:
    """Generate prototypes from clustered data.
    
    Args:
        vae_latents: VAE latents for all images.
        clip_img_embeds: CLIP image embeddings.
        clip_txt_embeds: CLIP text embeddings.
        descriptions: Text descriptions for each image.
        labels: Cluster labels for each image.
        ipc: Images per class (number of clusters).
        top_k_visual: Number of top visual samples to average.
        device: Device to run computations on.
        
    Returns:
        List of prototype dictionaries.
    """
    prototypes: List[Dict] = []
    
    for k in range(ipc):
        cluster_indices = np.where(labels == k)[0]
        if len(cluster_indices) == 0:
            continue
            
        cluster_vae_latents = vae_latents[cluster_indices]
        cluster_img_embeds = clip_img_embeds[cluster_indices]
        cluster_txt_embeds = clip_txt_embeds[cluster_indices]
        
        # Compute hubness scores
        k_visual = max(1, min(top_k_visual, len(cluster_indices)))
        
        if len(cluster_indices) == 1:
            top_local_indices = [0]
            best_idx_local = 0
        else:
            # Compute pairwise similarities
            sim = cluster_img_embeds @ cluster_img_embeds.t()
            n = cluster_img_embeds.size(0)
            row_sum = sim.sum(dim=1) - torch.ones(n, device=sim.device)
            avg_sim_local = row_sum / max(1, n - 1)
            
            top_local_indices = torch.topk(avg_sim_local, k_visual).indices.cpu().numpy()
            best_idx_local = int(torch.argmax(avg_sim_local).item())
        
        # Average top-k latents for image prototype
        top_latents = cluster_vae_latents[top_local_indices]
        image_prototype = torch.mean(top_latents, dim=0)
        
        # Select text prototype from best hubness sample
        text_prototype = descriptions[cluster_indices[best_idx_local]]
        
        # Compute average image-text similarity
        avg_sim = torch.mean(
            torch.sum(cluster_img_embeds * cluster_txt_embeds, dim=-1)
        ).item()
        
        prototypes.append({
            'image_prototype': image_prototype,
            'text_prototype': text_prototype,
            'cluster_size': len(cluster_indices),
            'avg_imgtxt_sim': avg_sim,
        })
    
    return prototypes


def aggregate_latents(
    latents: torch.Tensor,
    method: str = 'mean'
) -> torch.Tensor:
    """Aggregate multiple latents into a single prototype.
    
    Args:
        latents: Tensor of latents to aggregate [N, C, H, W].
        method: Aggregation method ('mean', 'median', 'first').
        
    Returns:
        Aggregated latent [C, H, W].
    """
    if method == 'mean':
        return latents.mean(dim=0)
    elif method == 'median':
        return latents.median(dim=0).values
    elif method == 'first':
        return latents[0]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

