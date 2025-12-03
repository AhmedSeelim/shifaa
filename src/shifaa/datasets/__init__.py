"""
Shifaa Datasets Module

Easy access to official Shifaa datasets hosted on HuggingFace.

Available Datasets:
- Shifaa Arabic Mental Health Consultations (35,648 consultations)
- Shifaa Arabic Medical Consultations (84,422 consultations)

Example:
    >>> from shifaa.datasets import load_shifaa_mental_dataset, load_shifaa_medical_dataset
    >>> mental_data = load_shifaa_mental_dataset()
    >>> medical_data = load_shifaa_medical_dataset()
"""

from .loader import (
    load_shifaa_mental_dataset,
    load_shifaa_medical_dataset,
    load_mental_health_dataset,
    load_medical_consultations_dataset,
    get_dataset_info,
    list_available_datasets,
    ShifaaDatasetConfig,
)

__all__ = [
    "load_shifaa_mental_dataset",
    "load_shifaa_medical_dataset",
    "load_mental_health_dataset",
    "load_medical_consultations_dataset",
    "get_dataset_info",
    "list_available_datasets",
    "ShifaaDatasetConfig",
]

__version__ = '0.1.0'