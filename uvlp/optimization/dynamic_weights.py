"""
Dynamic weight scheduling for Fisher objective components.

This module implements gap-based urgency weighting for balancing
intra-class, inter-class, and alignment metrics in the Fisher objective.
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np


def compute_dynamic_weights(
    intra_vals: List[float],
    inter_vals: List[float],
    align_vals: List[float],
    eps: float = 1e-8
) -> Tuple[float, float, float]:
    """Compute dynamic weights based on gap-based urgency.
    
    The weights are computed based on how far each metric is from its
    optimal value (min for intra, max for inter and align).
    
    Args:
        intra_vals: List of intra-class dispersion values across alpha probes.
        inter_vals: List of inter-class separation values across alpha probes.
        align_vals: List of text alignment values across alpha probes.
        eps: Small constant for numerical stability.
        
    Returns:
        Tuple of (w_intra, w_inter, w_align) normalized weights.
    """
    # Convert to numpy arrays
    intra = np.array(intra_vals)
    inter = np.array(inter_vals)
    align = np.array(align_vals)
    
    # Compute gaps (how much room for improvement)
    # For intra: gap = current - min (want to minimize)
    # For inter/align: gap = max - current (want to maximize)
    intra_gap = np.mean(intra) - np.min(intra) + eps
    inter_gap = np.max(inter) - np.mean(inter) + eps
    align_gap = np.max(align) - np.mean(align) + eps
    
    # Normalize to get weights (higher gap = higher weight)
    total = intra_gap + inter_gap + align_gap
    
    w_intra = intra_gap / total
    w_inter = inter_gap / total
    w_align = align_gap / total
    
    return float(w_intra), float(w_inter), float(w_align)


def compute_fisher_objective(
    intra: float,
    inter: float,
    align: float,
    w_intra: float = 1.0,
    w_inter: float = 1.0,
    w_align: float = 0.0
) -> float:
    """Compute the Fisher-style objective for alpha selection.
    
    The objective is: w_inter * inter + w_align * align - w_intra * intra
    
    Higher values are better (maximize inter-class separation and alignment,
    minimize intra-class dispersion).
    
    Args:
        intra: Intra-class dispersion (lower is better).
        inter: Inter-class separation (higher is better).
        align: Text alignment (higher is better).
        w_intra: Weight for intra-class term.
        w_inter: Weight for inter-class term.
        w_align: Weight for alignment term.
        
    Returns:
        Fisher objective value.
    """
    return w_inter * inter + w_align * align - w_intra * intra


def normalize_metrics(
    intra_vals: List[float],
    inter_vals: List[float],
    align_vals: List[float],
    eps: float = 1e-8
) -> Tuple[List[float], List[float], List[float]]:
    """Normalize metrics to [0, 1] range for fair comparison.
    
    Args:
        intra_vals: Raw intra-class values.
        inter_vals: Raw inter-class values.
        align_vals: Raw alignment values.
        eps: Small constant for numerical stability.
        
    Returns:
        Tuple of normalized (intra, inter, align) lists.
    """
    def normalize(vals):
        arr = np.array(vals)
        min_v, max_v = arr.min(), arr.max()
        if max_v - min_v < eps:
            return [0.5] * len(vals)
        return ((arr - min_v) / (max_v - min_v + eps)).tolist()
    
    return normalize(intra_vals), normalize(inter_vals), normalize(align_vals)

