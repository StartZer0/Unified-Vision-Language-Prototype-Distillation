"""
Visualization utilities for UVLP.

This module provides functions for plotting Fisher components, cluster metrics,
UMAP projections, and prototype grids.
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_fisher_components(
    probe_alphas: List[float],
    intra_vals: List[float],
    inter_vals: List[float],
    align_vals: List[float],
    save_dir: str,
    filename: str = 'fisher_components.png'
):
    """Plot Fisher components (intra, inter, align) vs alpha.
    
    Args:
        probe_alphas: List of alpha values probed.
        intra_vals: Intra-class dispersion values.
        inter_vals: Inter-class separation values.
        align_vals: Text alignment values.
        save_dir: Directory to save the plot.
        filename: Output filename.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(probe_alphas, intra_vals, label='Intra (lower is better)', marker='o')
    plt.plot(probe_alphas, inter_vals, label='Inter (higher is better)', marker='s')
    plt.plot(probe_alphas, align_vals, label='Align (higher is better)', marker='^')
    plt.title('Fisher Components vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Raw Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def plot_fisher_objective(
    probe_alphas: List[float],
    obj_vals: List[float],
    weights: tuple,
    save_dir: str,
    filename: str = 'fisher_objective.png'
):
    """Plot Fisher objective vs alpha.
    
    Args:
        probe_alphas: List of alpha values probed.
        obj_vals: Objective values at each alpha.
        weights: Tuple of (w_intra, w_inter, w_align) weights.
        save_dir: Directory to save the plot.
        filename: Output filename.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(probe_alphas, obj_vals, label='Objective', color='purple', marker='*')
    plt.title(f'Fisher Objective vs Alpha (Weights: {weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f})')
    plt.xlabel('Alpha')
    plt.ylabel('Objective Score')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def plot_cluster_metrics(
    prototypes: List[Dict],
    wnid: str,
    save_dir: str,
    filename: Optional[str] = None
):
    """Plot per-cluster metrics (alignment, alpha used).
    
    Args:
        prototypes: List of prototype dictionaries with metrics.
        wnid: Class WNID for labeling.
        save_dir: Directory to save the plot.
        filename: Output filename (default: {wnid}_cluster_metrics.png).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cluster_ids = list(range(len(prototypes)))
    aligns = [p.get('avg_imgtxt_sim', 0.0) for p in prototypes]
    alphas = [p.get('alpha_used', 0.0) for p in prototypes]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Alignment (Img-Txt)', color='tab:blue')
    ax1.bar(cluster_ids, aligns, color='tab:blue', alpha=0.6, label='Alignment')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Alpha Used', color='tab:red')
    ax2.plot(cluster_ids, alphas, color='tab:red', marker='o', label='Alpha')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title(f'Per-Cluster Metrics for {wnid}')
    
    if filename is None:
        filename = f'{wnid}_cluster_metrics.png'
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

