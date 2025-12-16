"""Distillation pipeline for UVLP."""

from uvlp.distillation.pipeline import run_jvl_distillation
from uvlp.distillation.prototype_generator import generate_prototypes
from uvlp.distillation.image_synthesis import synthesize_images, save_images_for_evaluation

__all__ = [
    "run_jvl_distillation",
    "generate_prototypes",
    "synthesize_images",
    "save_images_for_evaluation",
]

