#!/usr/bin/env python3
"""
Environment setup script for UVLP.

This script sets up the environment for running UVLP, including:
1. Patching the diffusers library with custom pipelines
2. Setting up cache directories
3. Downloading required models

Usage:
    python setup_environment.py --baseline-repo /path/to/vlcp
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import subprocess


def patch_diffusers(baseline_repo: str):
    """Patch the diffusers library with custom pipelines from VLCP.
    
    Args:
        baseline_repo: Path to the VLCP baseline repository.
    """
    print("Patching diffusers library...")
    
    # Find diffusers installation
    import diffusers
    diffusers_path = os.path.dirname(diffusers.__file__)
    pipelines_path = os.path.join(diffusers_path, 'pipelines')
    
    # Source files from baseline
    scripts_dir = os.path.join(baseline_repo, '02_diffusion_model_training', 'scripts')
    
    files_to_copy = [
        ('pipeline_stable_diffusion_latents2img.py', 
         os.path.join(pipelines_path, 'stable_diffusion', 'pipeline_stable_diffusion_latents2img.py')),
    ]
    
    for src_name, dst_path in files_to_copy:
        src_path = os.path.join(scripts_dir, src_name)
        if os.path.isfile(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"  Copied: {src_name}")
        else:
            print(f"  Warning: {src_name} not found in {scripts_dir}")
    
    # Update __init__.py to include the new pipeline
    init_path = os.path.join(pipelines_path, 'stable_diffusion', '__init__.py')
    if os.path.isfile(init_path):
        with open(init_path, 'r') as f:
            content = f.read()
        
        if 'StableDiffusionLatents2ImgPipeline' not in content:
            # Add import
            import_line = 'from .pipeline_stable_diffusion_latents2img import StableDiffusionLatents2ImgPipeline'
            
            # Find a good place to add it
            if 'from .pipeline_stable_diffusion_img2img' in content:
                content = content.replace(
                    'from .pipeline_stable_diffusion_img2img',
                    f'{import_line}\nfrom .pipeline_stable_diffusion_img2img'
                )
            else:
                content += f'\n{import_line}\n'
            
            with open(init_path, 'w') as f:
                f.write(content)
            print("  Updated __init__.py")
    
    # Also update the main diffusers __init__.py
    main_init = os.path.join(pipelines_path, '__init__.py')
    if os.path.isfile(main_init):
        with open(main_init, 'r') as f:
            content = f.read()
        
        if 'StableDiffusionLatents2ImgPipeline' not in content:
            # Add to exports
            if '"StableDiffusionImg2ImgPipeline"' in content:
                content = content.replace(
                    '"StableDiffusionImg2ImgPipeline"',
                    '"StableDiffusionImg2ImgPipeline",\n        "StableDiffusionLatents2ImgPipeline"'
                )
                with open(main_init, 'w') as f:
                    f.write(content)
                print("  Updated main __init__.py")
    
    print("Diffusers patching complete!")


def setup_cache_dirs(base_dir: str = None):
    """Set up cache directories for models.
    
    Args:
        base_dir: Base directory for caches.
    """
    if base_dir is None:
        base_dir = os.path.expanduser('~/.cache/uvlp')
    
    os.makedirs(base_dir, exist_ok=True)
    
    os.environ['HF_HOME'] = os.path.join(base_dir, 'huggingface')
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(base_dir, 'transformers')
    os.environ['DIFFUSERS_CACHE'] = os.path.join(base_dir, 'diffusers')
    os.environ['TORCH_HOME'] = os.path.join(base_dir, 'torch')
    
    print(f"Cache directories set up in: {base_dir}")


def main():
    parser = argparse.ArgumentParser(description='Setup UVLP environment')
    parser.add_argument(
        '--baseline-repo', type=str, required=True,
        help='Path to VLCP baseline repository'
    )
    parser.add_argument(
        '--cache-dir', type=str, default=None,
        help='Base directory for model caches'
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.baseline_repo):
        print(f"Error: Baseline repo not found: {args.baseline_repo}")
        sys.exit(1)
    
    print("Setting up UVLP environment...")
    print("=" * 50)
    
    setup_cache_dirs(args.cache_dir)
    patch_diffusers(args.baseline_repo)
    
    print("=" * 50)
    print("Setup complete!")
    print("\nYou can now run distillation with:")
    print("  python scripts/run_distillation.py --dataset imagewoof --ipc 10")


if __name__ == '__main__':
    main()

