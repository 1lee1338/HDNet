"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .inria import InriaSegmentation
from .whu_aerial import AerialSegmentation
from .whu_satellite import Whu_satelliteSegmentation

datasets = {
    'inria':InriaSegmentation,
    'whu_aerial':AerialSegmentation,
    'whu_satellite':Whu_satelliteSegmentation,
}

def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
