"""
Memory management utilities.

This module provides functions for managing GPU memory and clearing caches
during the distillation pipeline.
"""

from __future__ import annotations

import gc
from typing import Any

import torch


def free_memory(*objs: Any):
    """Release references and clear Python/Torch caches without raising on failures.
    
    Args:
        *objs: Objects to delete and release from memory.
    """
    for o in objs:
        try:
            del o
        except Exception:
            pass
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_memory_stats() -> dict:
    """Get current GPU memory statistics.
    
    Returns:
        Dictionary with memory allocation and cache information.
    """
    if not torch.cuda.is_available():
        return {'cuda_available': False}
    
    return {
        'cuda_available': True,
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
    }


def clear_all_caches():
    """Aggressively clear all Python and CUDA caches."""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def set_memory_efficient_mode():
    """Configure PyTorch for memory-efficient operation."""
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'

