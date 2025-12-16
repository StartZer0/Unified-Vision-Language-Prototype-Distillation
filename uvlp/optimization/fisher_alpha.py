"""
Fisher-style alpha selection on the unit sphere.

This module implements the global alpha selection algorithm that maximizes
a Fisher-style objective combining intra-class dispersion, inter-class
separation, and text alignment metrics.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPProcessor

from uvlp.optimization.golden_section import golden_section_maximize
from uvlp.utils.visualization import plot_fisher_components, plot_fisher_objective


def class_stats_for_alpha(
    alpha: float,
    img_feats_by_class: Dict[str, torch.Tensor],
    txt_feats_by_class: Dict[str, torch.Tensor],
    name_tf_by_class: Dict[str, torch.Tensor],
    device: str,
) -> Tuple[float, float, float]:
    """Compute Fisher-style intra/inter/alignment statistics for a given alpha.
    
    Args:
        alpha: Fusion weight (0=text only, 1=image only).
        img_feats_by_class: Dictionary mapping WNID to image feature tensors.
        txt_feats_by_class: Dictionary mapping WNID to text feature tensors.
        name_tf_by_class: Dictionary mapping WNID to class name text features.
        device: Device to run computations on.
        
    Returns:
        Tuple of (avg_intra, inter_sep, align_mean).
    """
    mu_by_class: Dict[str, torch.Tensor] = {}
    intra_vals: List[float] = []
    align_vals: List[float] = []

    for wnid, img_feats in img_feats_by_class.items():
        if img_feats.numel() == 0:
            continue
        txt_feats = txt_feats_by_class[wnid].to(device)
        img_feats = img_feats.to(device)
        fused = alpha * img_feats + (1.0 - alpha) * txt_feats
        fused = fused / fused.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        mu = fused.mean(dim=0, keepdim=True)
        mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = (fused @ mu.t()).squeeze(-1)
        intra = (1.0 - cos).mean().item()
        intra_vals.append(intra)
        mu_by_class[wnid] = mu.squeeze(0)

        tau = name_tf_by_class[wnid].to(device)
        align_vals.append(float(torch.dot(mu.squeeze(0), tau).item()))

    wnids = list(mu_by_class.keys())
    margins: List[float] = []
    for i, wi in enumerate(wnids):
        mui = mu_by_class[wi]
        others = [mu_by_class[wj] for j, wj in enumerate(wnids) if j != i]
        if not others:
            continue
        others_tensor = torch.stack(others, dim=0)
        sims = others_tensor @ mui
        nearest = float(torch.max(sims).item())
        margins.append(1.0 - nearest)

    avg_intra = float(np.mean(intra_vals)) if intra_vals else float('nan')
    inter_sep = float(np.mean(margins)) if margins else float('nan')
    align_mean = float(np.mean(align_vals)) if align_vals else float('nan')
    return avg_intra, inter_sep, align_mean


def _robust_fit(values: List[float]) -> Tuple[float, float]:
    """Compute robust median and MAD for normalization."""
    arr = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 1e-12:
        mad = 1e-12
    return med, mad


def _robust_norm(val: float, med: float, mad: float) -> float:
    """Normalize value using robust z-score."""
    if math.isnan(val):
        return 0.5
    z = (val - med) / (1.4826 * mad)
    return float(np.clip(0.5 + 0.1 * z, 0.0, 1.0))


def select_alpha_fisher_on_sphere(
    img_feats_by_class: Dict[str, torch.Tensor],
    txt_feats_by_class: Dict[str, torch.Tensor],
    name_tf_by_class: Dict[str, torch.Tensor],
    device: str,
    a_min: float = 0.2,
    a_max: float = 0.8,
    weights: Tuple[float, float, float] = (0.25, 0.50, 0.25),
    save_dir: Optional[str] = None,
) -> float:
    """Global alpha selection maximizing Fisher-style objective on the unit sphere.
    
    Args:
        img_feats_by_class: Dictionary mapping WNID to image feature tensors.
        txt_feats_by_class: Dictionary mapping WNID to text feature tensors.
        name_tf_by_class: Dictionary mapping WNID to class name text features.
        device: Device to run computations on.
        a_min: Minimum alpha value to search.
        a_max: Maximum alpha value to search.
        weights: Initial weights (w_inter, w_intra, w_align).
        save_dir: Directory to save diagnostic plots.
        
    Returns:
        Optimal alpha value.
    """
    # Probe alpha values
    probe = np.linspace(a_min, a_max, num=7).tolist()
    probe_stats = [
        class_stats_for_alpha(a, img_feats_by_class, txt_feats_by_class, name_tf_by_class, device)
        for a in probe
    ]
    intra_vals = [stats[0] for stats in probe_stats]
    inter_vals = [stats[1] for stats in probe_stats]
    align_vals = [stats[2] for stats in probe_stats]

    # Robust normalization parameters
    intra_med, intra_mad = _robust_fit(intra_vals)
    inter_med, inter_mad = _robust_fit(inter_vals)
    align_med, align_mad = _robust_fit(align_vals)

    # Dynamic weight scheduling (gap-based urgency)
    gap_intra = max(0.0, intra_med) * 3.0  # Urgency boosted x3
    gap_inter = max(0.0, 1.0 - inter_med)
    gap_align = max(0.0, 1.0 - align_med)

    print(f"[Dynamic Weight] Gaps: Intra={gap_intra:.3f}, Inter={gap_inter:.3f}, Align={gap_align:.3f}")

    w_inter_raw = weights[0] * gap_inter
    w_intra_raw = weights[1] * gap_intra
    w_align_raw = weights[2] * gap_align

    total_w = w_inter_raw + w_intra_raw + w_align_raw
    if total_w < 1e-6:
        total_w = 1.0
        w_inter_raw, w_intra_raw, w_align_raw = weights

    w_inter = w_inter_raw / total_w
    w_intra = w_intra_raw / total_w
    w_align = w_align_raw / total_w

    print(f"[Dynamic Weight] Final: Intra={w_intra:.3f}, Inter={w_inter:.3f}, Align={w_align:.3f}")

    def objective(a: float) -> float:
        intra, inter, align = class_stats_for_alpha(
            a, img_feats_by_class, txt_feats_by_class, name_tf_by_class, device
        )
        intra_n = _robust_norm(intra, intra_med, intra_mad)
        inter_n = _robust_norm(inter, inter_med, inter_mad)
        align_n = _robust_norm(align, align_med, align_mad)
        return (w_inter * inter_n) - (w_intra * intra_n) + (w_align * align_n)

    # Save diagnostic plots if requested
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            plot_fisher_components(probe, intra_vals, inter_vals, align_vals, save_dir)
            obj_vals = [objective(a) for a in probe]
            plot_fisher_objective(probe, obj_vals, (w_intra, w_inter, w_align), save_dir)
            print(f"Saved Fisher plots to {save_dir}")
        except Exception as e:
            print(f"Failed to save Fisher plots: {e}")

    # Golden section search for optimal alpha
    alpha_star, score = golden_section_maximize(objective, a_min, a_max, tol=1e-3)

    # Local refinement around alpha_star
    local_candidates = np.clip(
        np.array([
            alpha_star - 0.05,
            alpha_star - 0.025,
            alpha_star,
            alpha_star + 0.025,
            alpha_star + 0.05,
        ], dtype=float),
        a_min,
        a_max,
    )
    local_candidates = np.unique(local_candidates)
    best_alpha_local = float(alpha_star)
    best_intra_local = float('inf')

    for cand in local_candidates:
        intra_c, inter_c, align_c = class_stats_for_alpha(
            float(cand), img_feats_by_class, txt_feats_by_class, name_tf_by_class, device
        )
        if intra_c < best_intra_local:
            best_intra_local = intra_c
            best_alpha_local = float(cand)

    if not math.isclose(best_alpha_local, alpha_star, rel_tol=1e-4, abs_tol=1e-4):
        print(
            f"[Fisher alpha] local intra refine: {alpha_star:.4f} -> {best_alpha_local:.4f} "
            f"(intra {best_intra_local:.4f})"
        )

    final_obj = objective(best_alpha_local)
    print(f"[Fisher alpha] final alpha={best_alpha_local:.4f}, objective={final_obj:.4f}")
    return float(best_alpha_local)


def refine_alpha_per_class(
    wnid: str,
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    name_tf: torch.Tensor,
    global_alpha: float,
    device: str,
    delta: float = 0.1,
) -> float:
    """Refine alpha for a specific class around the global optimum.

    Args:
        wnid: Class WNID.
        img_feats: Image features for this class.
        txt_feats: Text features for this class.
        name_tf: Class name text feature.
        global_alpha: Global alpha from Fisher selection.
        device: Device to run computations on.
        delta: Search range around global alpha.

    Returns:
        Refined alpha for this class.
    """
    if img_feats.numel() == 0:
        return global_alpha

    a_min = max(0.0, global_alpha - delta)
    a_max = min(1.0, global_alpha + delta)

    def local_objective(a: float) -> float:
        fused = a * img_feats + (1.0 - a) * txt_feats
        fused = fused / fused.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        mu = fused.mean(dim=0, keepdim=True)
        mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # Intra-class dispersion
        cos = (fused @ mu.t()).squeeze(-1)
        intra = (1.0 - cos).mean().item()

        # Alignment with class name
        align = float(torch.dot(mu.squeeze(0), name_tf).item())

        # Objective: maximize alignment, minimize intra
        return align - intra

    alpha_refined, _ = golden_section_maximize(local_objective, a_min, a_max, tol=1e-3)
    return float(alpha_refined)

