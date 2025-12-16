"""
Dataset loading utilities for UVLP.

This module provides functions for loading ImageWoof, ImageNette, and ImageIDC
datasets with metadata and text descriptions.
"""

from __future__ import annotations

import os
import json
from typing import Dict, Tuple, List

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# ImageWoof class mapping (WNID -> Human-readable name)
IMAGEWOOF_CLASS_MAPPING = {
    'n02086240': 'Shih-Tzu',
    'n02087394': 'Rhodesian ridgeback',
    'n02088364': 'Beagle',
    'n02089973': 'English foxhound',
    'n02093754': 'Australian terrier',
    'n02096294': 'Border terrier',
    'n02099601': 'Golden retriever',
    'n02105641': 'Old English sheepdog',
    'n02111889': 'Samoyed',
    'n02115641': 'Dingo'
}

# ImageNette class mapping
IMAGENETTE_CLASS_MAPPING = {
    'n01440764': 'tench',
    'n02102040': 'English springer',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute'
}


class IndexedImageFolder(datasets.ImageFolder):
    """ImageFolder that returns (image, label, index) tuples."""
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, index


def load_text_descriptions(metadata_file_path: str) -> Dict[str, str]:
    """Read metadata.jsonl-style file into a mapping of relative path -> text prompt.
    
    Args:
        metadata_file_path: Path to the metadata.jsonl file.
        
    Returns:
        Dictionary mapping file paths to text descriptions.
    """
    descriptions: Dict[str, str] = {}
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            file_name = data.get('file_name', '')
            text = data.get('text', '')
            descriptions[file_name] = text
    print(f'Loaded {len(descriptions)} text descriptions from {metadata_file_path}')
    return descriptions


def get_transform_protocol(img_size: int) -> transforms.Compose:
    """Basic resize + tensor transform matching the ImageWoof preprocessing protocol.
    
    Args:
        img_size: Target image size.
        
    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def create_imagewoof_dataloader(
    data_dir: str,
    transform: transforms.Compose,
    batch_size: int = 10,
    num_workers: int = 4
) -> Tuple[DataLoader, List[str], List[str]]:
    """Create an indexed ImageFolder dataloader for the ImageWoof train split.
    
    Args:
        data_dir: Root directory containing train/ subfolder.
        transform: Transform to apply to images.
        batch_size: Batch size for the dataloader.
        num_workers: Number of worker processes.
        
    Returns:
        Tuple of (dataloader, list of all paths, list of class names).
    """
    train_dir = os.path.join(data_dir, 'train')
    dataset = IndexedImageFolder(root=train_dir, transform=transform)

    # Build kwargs to avoid passing prefetch_factor when num_workers == 0
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    if num_workers and num_workers > 0:
        dl_kwargs.update(dict(persistent_workers=False, prefetch_factor=2))

    dataloader = DataLoader(dataset, **dl_kwargs)
    path_all = [path for path, _ in dataset.samples]
    print(f'Dataloader: {len(dataset)} images, classes={dataset.classes}')
    return dataloader, path_all, dataset.classes


def get_class_mapping(dataset_name: str) -> Dict[str, str]:
    """Get the class mapping for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (imagewoof, imagenette, imageidc).
        
    Returns:
        Dictionary mapping WNIDs to human-readable class names.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == 'imagewoof' or dataset_name == 'woof':
        return IMAGEWOOF_CLASS_MAPPING
    elif dataset_name == 'imagenette' or dataset_name == 'nette':
        return IMAGENETTE_CLASS_MAPPING
    elif dataset_name == 'imageidc' or dataset_name == 'idc':
        # ImageIDC uses a subset of ImageNet classes
        return IMAGENETTE_CLASS_MAPPING  # Placeholder - update as needed
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

