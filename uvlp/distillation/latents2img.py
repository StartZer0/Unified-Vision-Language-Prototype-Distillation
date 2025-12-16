"""
Latents2Img pipeline loader for UVLP.

This module provides functions for loading the custom StableDiffusionLatents2ImgPipeline
from the patched diffusers installation.
"""

from __future__ import annotations

import torch


def load_latents2img_pipeline(model_path: str):
    """Load StableDiffusionLatents2ImgPipeline from the patched diffusers install.
    
    Args:
        model_path: Path to the fine-tuned Stable Diffusion model.
        
    Returns:
        Loaded and configured pipeline.
        
    Raises:
        ImportError: If the patched diffusers is not installed.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    try:
        from diffusers import StableDiffusionLatents2ImgPipeline  # type: ignore
    except Exception as e:
        raise ImportError(
            'StableDiffusionLatents2ImgPipeline not importable.\n'
            'Run setup_environment.py first to patch and reinstall diffusers.\n'
            'See the VLCP baseline repository for patching instructions.'
        ) from e

    pipe = StableDiffusionLatents2ImgPipeline.from_pretrained(
        model_path, 
        torch_dtype=dtype
    ).to(device)

    # Memory savers: identical outputs, lower VRAM
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    pipe.set_progress_bar_config(disable=True)
    print('Loaded Latents2Img from patched diffusers (memory-optimized)')
    return pipe


def configure_pipeline_for_inference(pipe, device: str = 'cuda'):
    """Configure pipeline for optimal inference performance.
    
    Args:
        pipe: The loaded pipeline.
        device: Target device.
        
    Returns:
        Configured pipeline.
    """
    # Faster on big GPUs (disable slicing)
    try:
        pipe.disable_attention_slicing()
    except Exception:
        pass

    # Try xFormers attention if available (identical outputs)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers memory efficient attention enabled")
    except Exception as e:
        print(f"xFormers not available: {e}")

    return pipe

