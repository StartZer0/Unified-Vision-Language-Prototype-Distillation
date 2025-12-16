"""
Spherical K-Means clustering for CLIP embeddings.

This module implements spherical K-Means clustering that operates on
L2-normalized data using cosine geometry, which is appropriate for
CLIP embeddings that lie on the unit hypersphere.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


def spherical_kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_init: int = 10,
    max_iter: int = 50,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spherical K-Means on L2-normalized data (cosine geometry).
    
    Args:
        X: Data matrix of shape (N, D), will be L2-normalized internally.
        n_clusters: Number of clusters to form.
        n_init: Number of random initializations to try.
        max_iter: Maximum iterations per initialization.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (centers, labels) where:
            - centers: Cluster centers of shape (n_clusters, D), unit L2-normalized.
            - labels: Cluster assignments of shape (N,).
    """
    assert n_clusters >= 1, "n_clusters must be >= 1"
    rng = np.random.RandomState(random_state)
    X = X.astype(np.float32, copy=True)
    
    # Normalize rows to unit length
    X_norm = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    X = X / X_norm

    N, D = X.shape
    best_inertia = np.inf
    best_centers = None
    best_labels = None

    for _ in range(max(n_init, 1)):
        # Initialize centers by random samples
        if N >= n_clusters:
            idx = rng.choice(N, size=n_clusters, replace=False)
        else:
            idx = np.arange(N)
        centers = X[idx].copy()
        # Ensure unit norm
        centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)

        labels = np.zeros(N, dtype=np.int32)
        for _it in range(max_iter):
            # Assignment by cosine similarity (argmax dot product)
            sims = X @ centers.T  # (N, K)
            new_labels = sims.argmax(axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # Update centers: mean then renormalize; handle empty clusters
            for k in range(n_clusters):
                mask = labels == k
                if not np.any(mask):
                    # Re-seed empty cluster to a random point
                    centers[k] = X[rng.randint(0, N)]
                else:
                    centers[k] = X[mask].mean(axis=0)
            centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)

        # Inertia under cosine distance: sum_i (1 - max_k cos(x_i, c_k))
        sims = X @ centers.T
        inertia = float(np.sum(1.0 - sims.max(axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()

    if best_centers is None:
        # Fallback single cluster
        best_centers = X.mean(axis=0, keepdims=True)
        best_centers /= np.linalg.norm(best_centers, axis=1, keepdims=True).clip(min=1e-8)
        best_labels = np.zeros(N, dtype=np.int32)
    
    return best_centers.astype(np.float32), best_labels.astype(np.int32)


def compute_cluster_quality(
    X: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray
) -> dict:
    """Compute quality metrics for spherical k-means clustering.
    
    Args:
        X: Data matrix of shape (N, D).
        centers: Cluster centers of shape (K, D).
        labels: Cluster assignments of shape (N,).
        
    Returns:
        Dictionary with quality metrics.
    """
    # Normalize
    X_norm = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    X = X / X_norm
    
    n_clusters = centers.shape[0]
    
    # Intra-cluster cohesion (average cosine similarity within clusters)
    intra_sims = []
    for k in range(n_clusters):
        mask = labels == k
        if np.sum(mask) > 0:
            cluster_points = X[mask]
            sims = cluster_points @ centers[k]
            intra_sims.append(np.mean(sims))
    
    # Inter-cluster separation (average pairwise distance between centers)
    if n_clusters > 1:
        center_sims = centers @ centers.T
        np.fill_diagonal(center_sims, 0)
        inter_sep = 1.0 - np.mean(center_sims[np.triu_indices(n_clusters, k=1)])
    else:
        inter_sep = 1.0
    
    return {
        'intra_cohesion': float(np.mean(intra_sims)) if intra_sims else 0.0,
        'inter_separation': float(inter_sep),
        'inertia': float(np.sum(1.0 - (X @ centers.T).max(axis=1))),
    }

