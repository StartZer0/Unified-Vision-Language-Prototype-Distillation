"""
Joint Vision-Language Clustering (JVL-C) for prototype generation.

This module implements the core JVLClustering class that performs
joint vision-language clustering in CLIP space for dataset distillation.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.neighbors import LocalOutlierFactor
from transformers import CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL

from uvlp.clustering.spherical_kmeans import spherical_kmeans
from uvlp.optimization.golden_section import golden_section_maximize


class JVLClustering:
    """Phase 1: Joint Vision-Language Clustering in CLIP space.
    
    This class implements the core JVL-C algorithm that:
    1. Extracts CLIP image and text embeddings
    2. Fuses them with an adaptive alpha parameter
    3. Performs spherical k-means clustering
    4. Selects representative prototypes using hubness-aware selection
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        device: str = 'cuda',
        diag_checks: bool = True,
        random_state: int = 0,
        alpha_optimize: bool = True,
        alpha_opt_min: float = 0.2,
        alpha_opt_max: float = 0.8,
        alpha_opt_steps: int = 13,
        alpha_reg: float = 0.6,
        top_k_visual: int = 3,
        sim_high_threshold: float = 0.85,
        sim_low_threshold: float = 0.65,
        alpha_when_high: float = 0.3,
        alpha_when_low: float = 0.7,
        clip_model_name: str = 'openai/clip-vit-large-patch14',
        sd_model_name: str = 'runwayml/stable-diffusion-v1-5',
    ):
        """Initialize JVL-C clustering.
        
        Args:
            alpha: Default fusion weight (0=text only, 1=image only).
            device: Device to run computations on.
            diag_checks: Whether to run diagnostic checks.
            random_state: Random seed for reproducibility.
            alpha_optimize: Whether to optimize alpha per cluster.
            alpha_opt_min: Minimum alpha for optimization.
            alpha_opt_max: Maximum alpha for optimization.
            alpha_opt_steps: Number of steps for alpha grid search.
            alpha_reg: Regularization strength for alpha optimization.
            top_k_visual: Number of top visual samples to average.
            sim_high_threshold: Threshold for high similarity.
            sim_low_threshold: Threshold for low similarity.
            alpha_when_high: Alpha to use when similarity is high.
            alpha_when_low: Alpha to use when similarity is low.
            clip_model_name: Name of CLIP model to use.
            sd_model_name: Name of Stable Diffusion model for VAE.
        """
        self.alpha = alpha
        self.device = device
        self.diag_checks = diag_checks
        self.random_state = random_state
        self.sim_high_threshold = sim_high_threshold
        self.sim_low_threshold = sim_low_threshold
        self.alpha_when_high = alpha_when_high
        self.alpha_when_low = alpha_when_low
        self.alpha_optimize = alpha_optimize
        self.alpha_opt_min = alpha_opt_min
        self.alpha_opt_max = alpha_opt_max
        self.alpha_opt_steps = alpha_opt_steps
        self.alpha_reg = alpha_reg
        self.top_k_visual = top_k_visual

        print('Loading CLIP model...')
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()

        print('Loading VAE model...')
        self.vae = AutoencoderKL.from_pretrained(
            sd_model_name,
            subfolder='vae',
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        )
        self.vae = self.vae.to(device)
        self.vae.eval()
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()

        print(f'JVL-C initialized (alpha={alpha})')

    def normalize(self, v: torch.Tensor) -> torch.Tensor:
        """L2-normalize embeddings along the last dimension."""
        return v / torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)

    def extract_vae_latents(self, images: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
        """Encode a batch of images into scaled VAE latents.
        
        Args:
            images: Batch of images as tensors [B, C, H, W] in [0, 1].
            batch_size: Batch size for processing.
            
        Returns:
            VAE latents of shape [B, 4, H/8, W/8].
        """
        latents_out = []
        with torch.inference_mode():
            for i in range(0, images.size(0), batch_size):
                chunk = images[i:i + batch_size]
                chunk = chunk.to(device=self.vae.device, dtype=self.vae.dtype, non_blocking=True)
                chunk = 2.0 * chunk - 1.0  # Scale to [-1, 1]
                dist = self.vae.encode(chunk).latent_dist
                lat = dist.sample()
                lat = lat * self.vae.config.scaling_factor
                latents_out.append(lat)
                del chunk, dist, lat
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return torch.cat(latents_out, dim=0)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents back to image space.
        
        Args:
            latents: VAE latents of shape [B, 4, H/8, W/8].
            
        Returns:
            Images of shape [B, 3, H, W] in [0, 1].
        """
        with torch.inference_mode():
            latents = latents.to(device=self.vae.device, dtype=self.vae.dtype, non_blocking=True)
            latents = latents / self.vae.config.scaling_factor
            images = self.vae.decode(latents).sample
            images = (images + 1.0) / 2.0
        return torch.clamp(images, 0.0, 1.0)

    def extract_clip_embeddings(
        self,
        images: torch.Tensor,
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized CLIP image and text embeddings.

        Args:
            images: Batch of images as tensors [B, C, H, W].
            texts: List of text descriptions.

        Returns:
            Tuple of (normalized image features, normalized text features).
        """
        with torch.no_grad():
            image_inputs = self.clip_processor(
                images=[transforms.ToPILImage()(img) for img in images],
                return_tensors='pt',
                padding=True,
            ).to(self.device)

            text_inputs = self.clip_processor(
                text=texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
            ).to(self.device)

            image_features = self.clip_model.get_image_features(**image_inputs)
            text_features = self.clip_model.get_text_features(**text_inputs)
            image_features = self.normalize(image_features)
            text_features = self.normalize(text_features)
        return image_features, text_features

    def create_fused_embeddings(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        """Blend image/text embeddings with alpha and renormalize.

        Args:
            image_features: Normalized image features.
            text_features: Normalized text features.
            alpha: Fusion weight (default: self.alpha).

        Returns:
            Fused and normalized embeddings.
        """
        a = self.alpha if alpha is None else alpha
        fused = a * image_features + (1 - a) * text_features
        return self.normalize(fused)

    def _optimize_alpha_for_cluster(
        self,
        img_feats_c: torch.Tensor,
        txt_feats_c: torch.Tensor,
        tau_pos: Optional[torch.Tensor] = None,
        tau_neg_stack: Optional[torch.Tensor] = None,
        alpha_prior: Optional[float] = None,
        lambda_prior: Optional[float] = None,
        w_intra: float = 1.0,
        w_align: float = 0.7,
        w_margin: float = 0.2,
        w_cluster_sep: float = 1.0,
        other_cluster_centers: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Directional Fisher objective for per-cluster alpha selection.

        Args:
            img_feats_c: Image features for this cluster.
            txt_feats_c: Text features for this cluster.
            tau_pos: Positive class text prototype.
            tau_neg_stack: Stack of negative class text prototypes.
            alpha_prior: Prior alpha value for regularization.
            lambda_prior: Regularization strength.
            w_intra: Weight for intra-cluster dispersion.
            w_align: Weight for text alignment.
            w_margin: Weight for margin to negative classes.
            w_cluster_sep: Weight for inter-cluster separation.
            other_cluster_centers: Centroids of other clusters in same class.

        Returns:
            Tuple of (optimal alpha, metrics dict).
        """
        device = img_feats_c.device
        a_min = float(self.alpha_opt_min)
        a_max = float(self.alpha_opt_max)
        lam = self.alpha_reg if lambda_prior is None else lambda_prior
        tau_pos = tau_pos.to(device) if tau_pos is not None else None
        if tau_neg_stack is not None:
            tau_neg_stack = tau_neg_stack.to(device)

        # Dynamic weighting probe
        probe_steps = 5
        probe_alphas = torch.linspace(a_min, a_max, steps=probe_steps).tolist()
        intra_vals, align_vals, margin_vals = [], [], []

        for a in probe_alphas:
            fused = a * img_feats_c + (1.0 - a) * txt_feats_c
            fused = fused / fused.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            mu = fused.mean(dim=0, keepdim=True)
            mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            intra = (1.0 - (fused @ mu.t()).squeeze(-1)).mean().item()
            align = float(torch.dot(mu.squeeze(0), tau_pos).item()) if tau_pos is not None else 0.0

            if tau_neg_stack is not None and tau_neg_stack.numel() > 0:
                sims = (mu @ tau_neg_stack.t()).squeeze(0)
                k = min(3, sims.numel())
                worst_neg = torch.topk(sims, k=k, largest=True).values.mean().item()
                margin = 1.0 - float(worst_neg)
            else:
                margin = 1.0

            intra_vals.append(intra)
            align_vals.append(align)
            margin_vals.append(margin)

        # Compute dynamic weights based on gaps
        intra_med = float(np.median(intra_vals))
        align_med = float(np.median(align_vals))
        margin_med = float(np.median(margin_vals))

        gap_intra = max(0.0, intra_med)
        gap_align = max(0.0, 1.0 - align_med)
        gap_margin = max(0.0, 1.0 - margin_med)

        w_intra_dyn = w_intra * gap_intra
        w_align_dyn = w_align * gap_align
        w_margin_dyn = w_margin * gap_margin

        total_w = w_intra_dyn + w_align_dyn + w_margin_dyn
        if total_w < 1e-6:
            w_intra_final, w_align_final, w_margin_final = w_intra, w_align, w_margin
        else:
            w_intra_final = w_intra_dyn / total_w
            w_align_final = w_align_dyn / total_w
            w_margin_final = w_margin_dyn / total_w

        # Inter-cluster separation weight
        w_cluster_sep_final = 0.0
        if other_cluster_centers is not None and other_cluster_centers.numel() > 0:
            other_cluster_centers = other_cluster_centers.to(device)
            gap_cluster_sep_vals = []
            m_div = 0.4
            for a in probe_alphas:
                fused = a * img_feats_c + (1.0 - a) * txt_feats_c
                fused = fused / fused.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                mu = fused.mean(dim=0, keepdim=True)
                mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                sims_to_others = (mu @ other_cluster_centers.t()).squeeze(0)
                worst_cluster_sim = sims_to_others.max().item() if sims_to_others.numel() > 0 else 0.0
                cluster_margin = 1.0 - worst_cluster_sim
                gap_cluster_sep_vals.append(max(0.0, m_div - min(cluster_margin, m_div)))

            gap_cluster_sep = float(np.median(gap_cluster_sep_vals))
            w_cluster_sep_dyn = w_cluster_sep * (gap_cluster_sep / m_div)

            total_w = w_intra_final + w_align_final + w_margin_final + w_cluster_sep_dyn
            if total_w > 1e-6:
                w_intra_final = w_intra_final / total_w
                w_align_final = w_align_final / total_w
                w_margin_final = w_margin_final / total_w
                w_cluster_sep_final = w_cluster_sep_dyn / total_w

        def cluster_objective(a: float) -> float:
            fused = a * img_feats_c + (1.0 - a) * txt_feats_c
            fused = fused / fused.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            mu = fused.mean(dim=0, keepdim=True)
            mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            cos = (fused @ mu.t()).squeeze(-1)
            intra = (1.0 - cos).mean().item()
            align = float(torch.dot(mu.squeeze(0), tau_pos).item()) if tau_pos is not None else 0.0

            if tau_neg_stack is not None and tau_neg_stack.numel() > 0:
                sims = (mu @ tau_neg_stack.t()).squeeze(0)
                k = min(3, sims.numel())
                worst_neg = torch.topk(sims, k=k, largest=True).values.mean().item()
                margin = 1.0 - float(worst_neg)
            else:
                margin = 0.0

            cluster_sep = 0.0
            if other_cluster_centers is not None and other_cluster_centers.numel() > 0:
                sims_to_others = (mu @ other_cluster_centers.t()).squeeze(0)
                worst_cluster_sim = sims_to_others.max().item() if sims_to_others.numel() > 0 else 0.0
                cluster_margin = 1.0 - worst_cluster_sim
                m_div = 0.4
                cluster_sep = min(cluster_margin, m_div)

            score = (-w_intra_final * intra) + (w_align_final * align) + (w_margin_final * margin) + (w_cluster_sep_final * cluster_sep)
            if alpha_prior is not None and lam > 0.0:
                score -= lam * ((a - alpha_prior) ** 2)
            return float(score)

        alpha_star, _ = golden_section_maximize(cluster_objective, a_min, a_max, tol=1e-3)

        # Compute final metrics at optimal alpha
        fused_star = alpha_star * img_feats_c + (1.0 - alpha_star) * txt_feats_c
        fused_star = fused_star / fused_star.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        mu_star = fused_star.mean(dim=0, keepdim=True)
        mu_star = mu_star / mu_star.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos_star = (fused_star @ mu_star.t()).squeeze(-1)
        intra_star = (1.0 - cos_star).mean().item()
        align_star = float(torch.dot(mu_star.squeeze(0), tau_pos).item()) if tau_pos is not None else float('nan')

        if tau_neg_stack is not None and tau_neg_stack.numel() > 0:
            sims = (mu_star @ tau_neg_stack.t()).squeeze(0)
            k = min(3, sims.numel())
            worst_neg = torch.topk(sims, k=k, largest=True).values.mean().item()
            margin_star = 1.0 - float(worst_neg)
        else:
            margin_star = float('nan')

        metrics = {"intra": intra_star, "align": align_star, "margin": margin_star}
        return float(alpha_star), metrics

    def derive_prototypes_jvl_c(
        self,
        vae_latents: torch.Tensor,
        clip_img_embeds: torch.Tensor,
        clip_txt_embeds: torch.Tensor,
        descriptions: List[str],
        ipc: int,
        contamination: float = 0.1,
        alpha_override: Optional[float] = None,
        tau_pos: Optional[torch.Tensor] = None,
        tau_neg_stack: Optional[torch.Tensor] = None,
        align_weight: float = 0.7,
        wnid: Optional[str] = None,
    ) -> Tuple[List[Dict], np.ndarray, torch.Tensor]:
        """Create JVL-C prototypes for a single class.

        Args:
            vae_latents: VAE latents for all images in the class.
            clip_img_embeds: CLIP image embeddings.
            clip_txt_embeds: CLIP text embeddings.
            descriptions: Text descriptions for each image.
            ipc: Images per class (number of prototypes to generate).
            contamination: LOF contamination parameter for outlier detection.
            alpha_override: Override the default alpha value.
            tau_pos: Positive class text prototype.
            tau_neg_stack: Stack of negative class text prototypes.
            align_weight: Weight for alignment in optimization.
            wnid: Class WNID for logging.

        Returns:
            Tuple of (prototypes list, inlier mask, text embeddings).
        """
        print(f'\nJVL-C: IPC={ipc}, contamination={contamination}')
        alpha_base = self.alpha if alpha_override is None else float(alpha_override)
        fused_embeds = self.create_fused_embeddings(clip_img_embeds, clip_txt_embeds, alpha=alpha_base)
        print(f'Fused embeddings: {tuple(fused_embeds.shape)}')

        # Outlier detection with LOF
        if contamination > 0:
            print(f'Outlier detection (LOF) with contamination={contamination}')
            clf = LocalOutlierFactor(n_neighbors=10, contamination=contamination)
            X = fused_embeds.detach().cpu().numpy()
            y_pred = clf.fit_predict(X)
            inliers = y_pred == 1
            print(f'Inliers: {np.sum(inliers)}/{len(inliers)}')
            fused_embeds = fused_embeds[inliers]
            vae_latents = vae_latents[inliers]
            clip_img_embeds = clip_img_embeds[inliers]
            clip_txt_embeds = clip_txt_embeds[inliers]
            descriptions = [descriptions[i] for i in range(len(descriptions)) if inliers[i]]
        else:
            inliers = np.ones(len(fused_embeds), dtype=bool)

        if len(fused_embeds) < ipc:
            print(f'Not enough samples after filtering: {len(fused_embeds)} < {ipc}. Reducing IPC.')
            ipc = len(fused_embeds)

        # Spherical k-means clustering
        centers_np, labels_np = spherical_kmeans(
            fused_embeds.detach().cpu().numpy(),
            n_clusters=int(ipc),
            n_init=10,
            max_iter=50,
            random_state=self.random_state,
        )
        cluster_centers = torch.tensor(centers_np, dtype=torch.float32, device=self.device)
        labels = labels_np

        # Pre-compute initial cluster centroids for inter-cluster separation
        initial_cluster_centroids = []
        for k in range(ipc):
            cluster_mask = labels == k
            if cluster_mask.sum() > 0:
                cluster_fused = fused_embeds[cluster_mask]
                centroid = cluster_fused.mean(dim=0)
                centroid = centroid / centroid.norm().clamp_min(1e-8)
                initial_cluster_centroids.append(centroid)
            else:
                initial_cluster_centroids.append(None)

        valid_centroids = [c for c in initial_cluster_centroids if c is not None]
        if len(valid_centroids) > 1:
            all_cluster_centroids = torch.stack(valid_centroids)
        else:
            all_cluster_centroids = None
        print(f"Computed {len(valid_centroids)} cluster centroids for inter-cluster separation")

        prototypes: List[Dict] = []
        for k in range(ipc):
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) == 0:
                print(f'Empty cluster {k}')
                continue
            cluster_vae_latents = vae_latents[cluster_indices]

            # Build other cluster centroids (exclude self)
            if all_cluster_centroids is not None and len(valid_centroids) > 1:
                other_centroids_list = [
                    initial_cluster_centroids[j]
                    for j in range(ipc)
                    if j != k and initial_cluster_centroids[j] is not None
                ]
                other_cluster_centers = torch.stack(other_centroids_list) if other_centroids_list else None
            else:
                other_cluster_centers = None

            # Per-cluster alpha optimization
            img_feats_c = clip_img_embeds[cluster_indices]
            txt_feats_c = clip_txt_embeds[cluster_indices]
            alpha_c, metrics = self._optimize_alpha_for_cluster(
                img_feats_c,
                txt_feats_c,
                tau_pos=tau_pos,
                tau_neg_stack=tau_neg_stack,
                alpha_prior=alpha_base,
                lambda_prior=self.alpha_reg,
                w_align=align_weight,
                other_cluster_centers=other_cluster_centers,
            )
            avg_sim = torch.mean(torch.sum(img_feats_c * txt_feats_c, dim=-1)).item()

            # Build adaptive fused space for selection
            cluster_fused_adapt = self.create_fused_embeddings(img_feats_c, txt_feats_c, alpha=alpha_c)

            # Hubness-based selection
            k_visual = max(1, min(self.top_k_visual, len(cluster_indices)))

            if cluster_fused_adapt.size(0) == 1:
                top_local_indices = torch.tensor([0], device=cluster_fused_adapt.device)
                best_idx_local = 0
                best_hub_score = 1.0
            else:
                sim = cluster_fused_adapt @ cluster_fused_adapt.t()
                n = cluster_fused_adapt.size(0)
                row_sum = sim.sum(dim=1) - torch.ones(n, device=sim.device)
                avg_sim_local = row_sum / max(1, n - 1)
                top_local_indices = torch.topk(avg_sim_local, k_visual).indices
                best_hub_score, best_idx_local = torch.max(avg_sim_local, dim=0)
                best_hub_score = best_hub_score.item()
                best_idx_local = int(best_idx_local.item())

            top_global_indices = cluster_indices[top_local_indices.detach().cpu().numpy()]
            top_latents = vae_latents[top_global_indices]
            image_prototype_Zc = torch.mean(top_latents, dim=0)

            best_global_idx = cluster_indices[best_idx_local]
            top1_visual_latent = vae_latents[best_global_idx]

            print(f"  Selected mean of top-{k_visual} visual prototypes (hubness-based)")

            # Select text prototype
            chosen_global_idx = int(cluster_indices[best_idx_local])
            text_prototype_Tc = descriptions[chosen_global_idx]
            print(
                f"  Selected text prototype (hubness avg_sim={best_hub_score:.4f}, "
                f"cluster_avg_imgtxt={avg_sim:.4f}, alpha={alpha_c:.2f})"
            )

            prototypes.append({
                'image_prototype': image_prototype_Zc,
                'top1_visual_latent': top1_visual_latent,
                'text_prototype': text_prototype_Tc,
                'cluster_size': int(len(cluster_indices)),
                'cluster_center': cluster_centers[k],
                'alpha_used': alpha_c,
                'avg_imgtxt_sim': avg_sim,
            })

        print(f'JVL-C complete: {len(prototypes)} prototypes')
        return prototypes, inliers, clip_txt_embeds
