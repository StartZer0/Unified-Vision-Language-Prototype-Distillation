"""
Hubness-aware prototype selection.

This module implements hubness-aware selection of representative prototypes
based on average cosine similarity within clusters.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch


def select_by_hubness(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    top_k: int = 6,
) -> np.ndarray:
    """Select top-K most representative samples from a cluster by hubness.
    
    Hubness is measured as the average cosine similarity to all other
    points in the same cluster. Higher hubness = more central/representative.
    
    Args:
        embeddings: All embeddings of shape (N, D).
        labels: Cluster assignments of shape (N,).
        cluster_id: Which cluster to select from.
        top_k: Number of samples to select.
        
    Returns:
        Indices of the top-K most representative samples.
    """
    mask = labels == cluster_id
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return np.array([], dtype=np.int64)
    
    if len(indices) <= top_k:
        return indices
    
    # Get cluster embeddings
    cluster_embs = embeddings[mask]
    
    # Normalize
    norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True).clip(min=1e-8)
    cluster_embs_norm = cluster_embs / norms
    
    # Compute pairwise cosine similarities
    sim_matrix = cluster_embs_norm @ cluster_embs_norm.T
    
    # Average similarity to all other points (hubness score)
    # Exclude self-similarity by setting diagonal to 0
    np.fill_diagonal(sim_matrix, 0)
    hubness_scores = sim_matrix.mean(axis=1)
    
    # Select top-K by hubness
    top_k_local = np.argsort(hubness_scores)[-top_k:][::-1]
    
    return indices[top_k_local]


def select_representative_latents(
    vae_latents: torch.Tensor,
    clip_embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    top_k: int = 6,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Select representative VAE latents based on CLIP embedding hubness.
    
    Args:
        vae_latents: VAE latent tensors of shape (N, C, H, W).
        clip_embeddings: CLIP embeddings of shape (N, D).
        labels: Cluster assignments of shape (N,).
        cluster_id: Which cluster to select from.
        top_k: Number of samples to select.
        
    Returns:
        Tuple of (selected VAE latents, selected CLIP embeddings).
    """
    selected_indices = select_by_hubness(clip_embeddings, labels, cluster_id, top_k)
    
    if len(selected_indices) == 0:
        return torch.empty(0), np.empty((0, clip_embeddings.shape[1]))
    
    selected_latents = vae_latents[selected_indices]
    selected_embeddings = clip_embeddings[selected_indices]
    
    return selected_latents, selected_embeddings


def compute_prototype_from_cluster(
    vae_latents: torch.Tensor,
    clip_embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    top_k: int = 6,
    aggregation: str = 'mean',
) -> Tuple[torch.Tensor, np.ndarray]:
    """Compute a prototype latent from a cluster using hubness-aware selection.
    
    Args:
        vae_latents: VAE latent tensors of shape (N, C, H, W).
        clip_embeddings: CLIP embeddings of shape (N, D).
        labels: Cluster assignments of shape (N,).
        cluster_id: Which cluster to compute prototype for.
        top_k: Number of samples to use for prototype.
        aggregation: How to aggregate ('mean' or 'median').
        
    Returns:
        Tuple of (prototype VAE latent, prototype CLIP embedding).
    """
    selected_latents, selected_embeddings = select_representative_latents(
        vae_latents, clip_embeddings, labels, cluster_id, top_k
    )
    
    if len(selected_latents) == 0:
        return torch.empty(0), np.empty(clip_embeddings.shape[1])
    
    if aggregation == 'mean':
        proto_latent = selected_latents.mean(dim=0)
        proto_embedding = selected_embeddings.mean(axis=0)
    elif aggregation == 'median':
        proto_latent = selected_latents.median(dim=0).values
        proto_embedding = np.median(selected_embeddings, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    # Normalize embedding
    proto_embedding = proto_embedding / np.linalg.norm(proto_embedding).clip(min=1e-8)
    
    return proto_latent, proto_embedding

