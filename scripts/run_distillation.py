#!/usr/bin/env python3
"""
Main entry point for UVLP distillation.

This script runs the full Unified Vision Language Prototype distillation
pipeline for dataset distillation.

Usage:
    python run_distillation.py --dataset imagewoof --ipc 10 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from uvlp.configs.base_config import Config
from uvlp.distillation.pipeline import run_jvl_distillation
from uvlp.utils.environment import resolve_baseline_repo, try_mount_drive, in_colab
from uvlp.evaluation.minimax_eval import posthoc_minimax_eval_from_distilled


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified Vision Language Prototype Distillation'
    )
    parser.add_argument(
        '--dataset', type=str, default='imagewoof',
        choices=['imagewoof', 'imagenette', 'imageidc'],
        help='Dataset to distill'
    )
    parser.add_argument(
        '--ipc', type=int, default=10,
        help='Images per class'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Default fusion weight (0=text, 1=image)'
    )
    parser.add_argument(
        '--project-root', type=str, default=None,
        help='Project root directory'
    )
    parser.add_argument(
        '--finetuned-model', type=str, default=None,
        help='Path to fine-tuned Stable Diffusion model'
    )
    parser.add_argument(
        '--run-eval', action='store_true',
        help='Run Minimax evaluation after distillation'
    )
    parser.add_argument(
        '--eval-repeat', type=int, default=3,
        help='Number of evaluation repeats'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Mount Google Drive if in Colab
    if in_colab():
        try_mount_drive()
    
    # Initialize configuration
    config = Config(base_seed=args.seed, project_root=args.project_root)
    config.ipc = args.ipc
    config.alpha = args.alpha
    config.update_for_dataset(args.dataset)
    
    print("=" * 60)
    print("Unified Vision Language Prototype Distillation")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"IPC: {args.ipc}")
    print(f"Seed: {args.seed}")
    print(f"Alpha: {args.alpha}")
    print(f"Device: {config.device}")
    print("=" * 60)
    
    # Resolve baseline repository
    try:
        baseline_repo = resolve_baseline_repo(config.base_path)
        print(f"Baseline repo: {baseline_repo}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease clone the VLCP baseline repository:")
        print("  git clone https://github.com/StartZero0/Dataset-Distillation-via-Vision-Language-Category-Prototype.git")
        sys.exit(1)
    
    # Run distillation
    print("\nStarting distillation...")
    jvl_c_root = run_jvl_distillation(
        config=config,
        baseline_repo=baseline_repo,
        finetuned_model_path=args.finetuned_model,
    )
    
    print(f"\nDistillation complete!")
    print(f"Output: {jvl_c_root}")
    
    # Run evaluation if requested
    if args.run_eval:
        print("\nRunning Minimax evaluation...")
        posthoc_minimax_eval_from_distilled(
            distilled_seed_root=config.distilled_root,
            baseline_repo=baseline_repo,
            real_root=config.val_dir,
            ipc=config.ipc,
            seed=config.current_seed,
            repeat=args.eval_repeat,
            tag=f'uvlp_{args.dataset}',
            spec=config.spec,
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

