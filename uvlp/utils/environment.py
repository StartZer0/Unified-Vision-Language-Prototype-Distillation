"""
Environment detection and setup utilities.

This module provides functions for detecting the execution environment
(Colab, local) and setting up required paths and dependencies.
"""

from __future__ import annotations

import os
from typing import Optional


# Baseline repository candidate names
BASELINE_CANDIDATES = [
    'Dataset-Distillation-via-Vision-Language-Category-Prototype',
    'Dataset-Distillation-via-Vision-Language-Category-Prototype-main',
]

# Default paths for Colab environment
DEFAULT_PROJECT_ROOT = "/content/jvl_vlpr"


def in_colab() -> bool:
    """Detect whether the code is running inside a Google Colab runtime."""
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def try_mount_drive():
    """Best-effort mount of Google Drive when executing in Colab."""
    if in_colab():
        try:
            from google.colab import drive  # type: ignore
            drive.mount('/content/drive', force_remount=True)
            print('Google Drive mounted')
        except Exception as e:
            print(f'Drive mount failed (continuing): {e}')


def resolve_baseline_repo(project_root: str) -> str:
    """Resolve the baseline repository path across env vars, project_root, CWD, and Colab defaults.
    
    Args:
        project_root: The project root directory to search in.
        
    Returns:
        Path to the baseline VLCP repository.
        
    Raises:
        FileNotFoundError: If the baseline repository cannot be found.
    """
    # 1) Environment override
    env_path = os.environ.get('BASELINE_REPO_PATH')
    if env_path and os.path.isdir(env_path):
        return env_path
    
    # 2) Under provided project_root
    for name in BASELINE_CANDIDATES:
        p = os.path.join(project_root, name)
        if os.path.isdir(p):
            return p
    
    # 3) Under current working directory
    for name in BASELINE_CANDIDATES:
        p = os.path.join(os.getcwd(), name)
        if os.path.isdir(p):
            return p
    
    # 4) Typical Colab project root fallback
    if in_colab():
        for name in BASELINE_CANDIDATES:
            p = os.path.join(DEFAULT_PROJECT_ROOT, name)
            if os.path.isdir(p):
                return p
    
    raise FileNotFoundError(
        'Baseline repo not found. Please place the repository folder in your project root.\n'
        f'Tried: {BASELINE_CANDIDATES} under {project_root}, CWD, and {DEFAULT_PROJECT_ROOT}'
    )


def setup_cache_directories(base_dir: str = '/content/drive/MyDrive/VLCPbase'):
    """Set up environment variables for cache directories (useful in Colab).
    
    Args:
        base_dir: Base directory for cache storage.
    """
    os.environ['HF_HOME'] = os.path.join(base_dir, '.cache/huggingface')
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(base_dir, '.cache/transformers')
    os.environ['DIFFUSERS_CACHE'] = os.path.join(base_dir, '.cache/diffusers')
    os.environ['HF_HUB_CACHE'] = os.environ['HF_HOME']
    os.environ['TORCH_HOME'] = os.path.join(base_dir, '.cache/torch')
    os.environ['NLTK_DATA'] = os.path.join(base_dir, 'nltk_data')
    os.environ['PIP_CACHE_DIR'] = os.path.join(base_dir, '.cache/pip')


def ensure_class_file(path: str, class_mapping: dict):
    """Write the default WNID list to disk if the class file is missing.
    
    Args:
        path: Path to write the class file.
        class_mapping: Dictionary mapping WNIDs to class names.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        wnids = list(class_mapping.keys())
        with open(path, 'w') as f:
            f.write('\n'.join(wnids) + '\n')
        print(f'Wrote class file: {path}')
    else:
        print(f'Using class file: {path}')

