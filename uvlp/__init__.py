"""
Unified Vision Language Prototype Distillation (UVLP)

A novel framework for dataset distillation that combines vision and language
modalities through Joint Vision-Language Clustering in CLIP space.

This package extends the VLCP (Vision Language Category Prototype) baseline
with advanced clustering, alpha optimization, and hubness-aware selection.
"""

__version__ = "1.0.0"
__author__ = "Eljan Mammadov"

from uvlp.configs.base_config import Config
from uvlp.clustering.jvl_clustering import JVLClustering
from uvlp.distillation.pipeline import run_jvl_distillation

__all__ = [
    "Config",
    "JVLClustering", 
    "run_jvl_distillation",
]

