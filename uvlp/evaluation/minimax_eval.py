"""
Minimax evaluation utilities for UVLP.

This module provides functions for running the Minimax evaluation protocol
from the VLCP baseline repository.
"""

from __future__ import annotations

import os
import sys
import subprocess
from typing import Optional


def ensure_minimax_deps():
    """Install lightweight dependencies needed for Minimax evaluation if missing."""
    missing = []
    try:
        import efficientnet_pytorch  # type: ignore
    except Exception:
        missing.append('efficientnet-pytorch')
    try:
        import timm  # type: ignore
    except Exception:
        missing.append('timm')
    if missing:
        print('Installing Minimax deps:', missing)
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q'] + missing, 
            check=False
        )


def run_minimax_eval(
    baseline_repo: str,
    syn_root: str,
    real_root: str,
    ipc: int,
    seed: int,
    repeat: int,
    tag: str,
    model_name: str,
    depth: int,
    spec: str = 'woof',
    nclass: int = 10,
):
    """Invoke the Minimax training script for a given model configuration.
    
    Args:
        baseline_repo: Path to the baseline VLCP repository.
        syn_root: Path to synthetic (distilled) images.
        real_root: Path to real validation images.
        ipc: Images per class.
        seed: Random seed.
        repeat: Number of evaluation repeats.
        tag: Experiment tag.
        model_name: Model architecture name.
        depth: Model depth.
        spec: Dataset specification (woof, nette, idc).
        nclass: Number of classes.
    """
    eval_dir = os.path.join(baseline_repo, '04_evaluation', 'Minimax')
    train_py = os.path.join(eval_dir, 'train.py')
    
    if not os.path.isfile(train_py):
        raise FileNotFoundError(
            f"Minimax train.py not found at {train_py}.\n"
            "Please ensure the baseline repository is correctly set up."
        )
    
    cmd = [
        sys.executable, 'train.py',
        '-d', 'imagenet',
        '--imagenet_dir', syn_root, real_root,
        '-n', model_name, '--depth', str(depth),
        '--nclass', str(nclass), '--norm_type', 'instance',
        '--ipc', str(ipc), '--repeat', str(repeat),
        '--spec', spec, '--seed', str(seed), '--tag', tag,
        '--slct_type', 'random', '--verbose',
    ]
    
    print(f"Running Minimax evaluation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=eval_dir)


def posthoc_minimax_eval_from_distilled(
    distilled_seed_root: str,
    baseline_repo: str,
    real_root: str,
    ipc: int,
    seed: int,
    repeat: int,
    tag: str,
    which: str = 'jvl_c',
    spec: str = 'woof',
):
    """Run Minimax using a local distilled seed folder built from JVL-C outputs.
    
    Args:
        distilled_seed_root: Root directory containing distilled images.
        baseline_repo: Path to the baseline VLCP repository.
        real_root: Path to real validation images.
        ipc: Images per class.
        seed: Random seed.
        repeat: Number of evaluation repeats.
        tag: Experiment tag.
        which: Which distillation phase to evaluate ('jvl_c').
        spec: Dataset specification.
    """
    ensure_minimax_deps()
    
    if which.lower() != 'jvl_c':
        raise ValueError("Only 'jvl_c' synthetic outputs are available in this pipeline.")
    
    syn_root = os.path.join(distilled_seed_root, 'distilled_images_jvl_c')
    if not os.path.isdir(syn_root):
        raise FileNotFoundError(f'Synthetic root not found: {syn_root}')
    if not os.path.isdir(real_root):
        raise FileNotFoundError(f'Real root not found: {real_root}')

    # Run evaluation with multiple architectures
    run_minimax_eval(
        baseline_repo=baseline_repo, syn_root=syn_root, real_root=real_root,
        ipc=ipc, seed=seed, repeat=repeat, tag=f'{tag}_resnet18', 
        model_name='resnet', depth=18, spec=spec
    )
    run_minimax_eval(
        baseline_repo=baseline_repo, syn_root=syn_root, real_root=real_root,
        ipc=ipc, seed=seed, repeat=repeat, tag=f'{tag}_resnet_ap', 
        model_name='resnet_ap', depth=10, spec=spec
    )
    run_minimax_eval(
        baseline_repo=baseline_repo, syn_root=syn_root, real_root=real_root,
        ipc=ipc, seed=seed, repeat=repeat, tag=f'{tag}_convnet', 
        model_name='convnet', depth=6, spec=spec
    )

