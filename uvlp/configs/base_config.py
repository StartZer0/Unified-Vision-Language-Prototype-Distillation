"""
Configuration class for UVLP experiments.

This module contains the Config class that holds all hyperparameters and
filesystem paths for the Unified Vision Language Prototype Distillation framework.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from uvlp.utils.environment import in_colab


# Project naming constants
_PROJECT_SUFFIX = 'vlpr'
PROJECT_BASENAME = f"jvl_{_PROJECT_SUFFIX}"
DEFAULT_PROJECT_ROOT = f"/content/{PROJECT_BASENAME}"
DEFAULT_DRIVE_MODELS_ROOT = f"/content/drive/MyDrive/{PROJECT_BASENAME}_models"


class Config:
    """Container for experiment hyperparameters and resolved filesystem locations."""
    
    def __init__(self, base_seed: int = 0, project_root: Optional[str] = None):
        self.current_seed = base_seed

        # Dataset / protocol
        self.dataset = 'imagenet'
        self.spec = 'woof'
        self.nclass = 10
        self.resolution = 512
        self.output_image_size = 256  # saved image size for evaluation

        # Diffusion training (match baseline protocol)
        self.model_name = 'runwayml/stable-diffusion-v1-5'
        self.train_batch_size = 32
        self.gradient_accumulation_steps = 4
        self.mixed_precision = 'fp16'
        self.num_train_epochs = 8
        self.validation_epochs = 2
        self.learning_rate = 1e-5
        self.max_grad_norm = 1.0
        self.lr_scheduler = 'constant'
        self.lr_warmup_steps = 0
        self.use_ema = True

        # Prototype generation (IPC) and synthesis
        self.ipc = 10
        self.km_expand = 1
        self.contamination = 0.1
        self.guidance_scale = 10.5
        self.strength = 0.75
        self.negative_prompt = 'cartoon, anime, painting'

        # JVL-C params
        self.alpha = 0.5
        self.top_k_visual = 6  # Mean top-K representative visual latents

        # JVL-C alpha search controls
        self.jvlc_global_alpha_search = False
        self.jvlc_global_alpha_min = 0.2
        self.jvlc_global_alpha_max = 0.8
        self.jvlc_global_alpha_steps = 7
        self.jvlc_alpha_align_weight = 0.0

        # CLIP model
        self.clip_model_name = 'openai/clip-vit-large-patch14'

        # Evaluation
        self.eval_repeat = 3

        # Paths
        self.base_path = project_root or (
            DEFAULT_PROJECT_ROOT if in_colab() else os.getcwd()
        )
        
        # Data and outputs
        self.data_root = "/content/woofdata_local" if in_colab() else os.path.join(self.base_path, "data")
        self.train_dir = os.path.join(self.data_root, "train")
        self.val_dir = os.path.join(self.data_root, "val")

        self.outputs_root = os.path.join(self.base_path, "outputs")
        self.distilled_root = os.path.join(self.base_path, "distilled_images", f"seed_{self.current_seed}")
        self.prototype_root = os.path.join(self.base_path, "prototypes", f"seed_{self.current_seed}")

        # Class file path
        self.class_file = os.path.join(
            self.base_path,
            "Dataset-Distillation-via-Vision-Language-Category-Prototype",
            "03_distiilation", "label-prompt", "imagenet_woof_classes.txt"
        )

        self.metadata_file = os.path.join(self.train_dir, 'metadata.jsonl')

        # Fine-tuned model output
        models_dir_name = f"{PROJECT_BASENAME}_models"
        self.finetuned_model_path = (
            os.path.join(DEFAULT_DRIVE_MODELS_ROOT, 'ImageWoof_seed0')
            if in_colab()
            else os.path.join(self.base_path, models_dir_name, 'ImageWoof_seed0')
        )

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def set_seed(self, s: int):
        """Update the active seed and regenerate any seed-scoped output paths."""
        self.current_seed = s
        self.distilled_root = os.path.join(self.base_path, 'distilled_images', f'seed_{s}')
        self.prototype_root = os.path.join(self.base_path, 'prototypes', f'seed_{s}')

    def update_for_dataset(self, dataset_name: str):
        """Update configuration for a specific dataset (imagenette, imagewoof, imageidc)."""
        dataset_name = dataset_name.lower()
        if dataset_name == 'imagenette':
            self.spec = 'nette'
            self.nclass = 10
        elif dataset_name == 'imagewoof':
            self.spec = 'woof'
            self.nclass = 10
        elif dataset_name == 'imageidc':
            self.spec = 'idc'
            self.nclass = 10
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

