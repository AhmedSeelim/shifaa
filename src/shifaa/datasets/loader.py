"""
Shifaa Datasets Loader Module

Provides easy access to official Shifaa datasets hosted on HuggingFace.
"""

from typing import Optional, Dict, Any, Union
from datasets import load_dataset, Dataset, DatasetDict
import logging

# Library logger - do NOT configure global logging in a library. Users configure logging.
logger = logging.getLogger(__name__)


class ShifaaDatasetConfig:
    """Configuration for Shifaa datasets on HuggingFace."""

    MENTAL_HEALTH_REPO = "Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations"
    MEDICAL_CONSULTATIONS_REPO = "Ahmed-Selem/Shifaa_Arabic_Medical_Consultations"

    MENTAL_HEALTH_INFO = {
        "name": "Shifaa Arabic Mental Health Consultations",
        "size": 35_648,
        "specializations": 7,
        "diagnoses": 123,
        "language": "Arabic",
    }

    MEDICAL_CONSULTATIONS_INFO = {
        "name": "Shifaa Arabic Medical Consultations",
        "size": 84_422,
        "specializations": 16,
        "diagnoses": 585,
        "language": "Arabic",
        "avg_answer_length": 273,
    }


def _choose_split_from_dict(ds: DatasetDict, preferred: str = "train", split: Optional[str] = None) -> str:
    """
    Return chosen split name from a DatasetDict.
    If 'split' is provided and exists -> return it.
    Else return 'preferred' if exists, otherwise the first available split.
    """
    available = list(ds.keys())
    if split:
        if split in ds:
            return split
        raise ValueError(
            f"Requested split '{split}' not found. Available splits: {available}. "
            "Please choose one of the available splits or omit 'split' to use the default."
        )
    if preferred in ds:
        return preferred
    return available[0]


def _post_load_handle(dataset: Union[Dataset, DatasetDict],
                      split: Optional[str],
                      prefer_split: str,
                      return_dataset_dict: bool) -> Union[Dataset, DatasetDict]:
    """
    If dataset is DatasetDict and return_dataset_dict is False, return a single Dataset (preferred split).
    Otherwise return the DatasetDict.
    """
    if isinstance(dataset, Dataset):
        return dataset

    # It's a DatasetDict
    if return_dataset_dict:
        return dataset

    chosen = _choose_split_from_dict(dataset, preferred=prefer_split, split=split)
    return dataset[chosen]


def load_shifaa_mental_dataset(
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    *,
    prefer_split: str = "train",
    return_dataset_dict: bool = False,
    **kwargs
) -> Union[Dataset, DatasetDict]:
    """
    Load the Shifaa Arabic Mental Health Consultations dataset.

    By default this function returns a single `Dataset` object:
      - If the remote repo has splits and you do not pass `split`, it returns the `prefer_split` (default "train")
      - If you want the raw DatasetDict, set `return_dataset_dict=True`

    Args:
        split: explicit split name to load (e.g. "train"). If provided, must exist in the remote dataset.
        cache_dir: optional cache directory for huggingface datasets.
        prefer_split: which split to prefer when not specifying `split`.
        return_dataset_dict: if True, return the full DatasetDict instead of a single Dataset.
        **kwargs: forwarded to `datasets.load_dataset()`

    Returns:
        Dataset or DatasetDict
    """
    try:
        logger.info("Loading Shifaa Mental Health Dataset from HuggingFace...")
        logger.info("Repository: %s", ShifaaDatasetConfig.MENTAL_HEALTH_REPO)

        ds = load_dataset(
            ShifaaDatasetConfig.MENTAL_HEALTH_REPO,
            split=None if return_dataset_dict or split is None else split,
            cache_dir=cache_dir,
            **kwargs
        )

        # When split parameter passed to load_dataset it may return a Dataset directly.
        result = _post_load_handle(ds, split=split, prefer_split=prefer_split, return_dataset_dict=return_dataset_dict)

        if isinstance(result, Dataset):
            logger.info("✓ Successfully loaded mental health dataset — size: %d", len(result))
        else:
            logger.info("✓ Successfully loaded mental health dataset — splits: %s", list(result.keys()))

        return result

    except Exception as e:
        logger.exception("Failed to load mental health dataset")
        raise


def load_shifaa_medical_dataset(
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    *,
    prefer_split: str = "train",
    return_dataset_dict: bool = False,
    **kwargs
) -> Union[Dataset, DatasetDict]:
    """
    Load the Shifaa Arabic Medical Consultations dataset.

    Behavior mirrors `load_shifaa_mental_dataset()` regarding splits and return types.

    Args:
        split: explicit split name to load. If provided, must exist.
        cache_dir: optional cache directory.
        prefer_split: which split to prefer when not specifying `split`.
        return_dataset_dict: if True, return the raw DatasetDict.
        **kwargs: forwarded to `datasets.load_dataset()`

    Returns:
        Dataset or DatasetDict
    """
    try:
        logger.info("Loading Shifaa Medical Consultations Dataset from HuggingFace...")
        logger.info("Repository: %s", ShifaaDatasetConfig.MEDICAL_CONSULTATIONS_REPO)

        ds = load_dataset(
            ShifaaDatasetConfig.MEDICAL_CONSULTATIONS_REPO,
            split=None if return_dataset_dict or split is None else split,
            cache_dir=cache_dir,
            **kwargs
        )

        result = _post_load_handle(ds, split=split, prefer_split=prefer_split, return_dataset_dict=return_dataset_dict)

        if isinstance(result, Dataset):
            logger.info("✓ Successfully loaded medical consultations dataset — size: %d", len(result))
        else:
            logger.info("✓ Successfully loaded medical consultations dataset — splits: %s", list(result.keys()))

        return result

    except Exception as e:
        logger.exception("Failed to load medical consultations dataset")
        raise


def get_dataset_info(dataset_type: str = "mental") -> Dict[str, Any]:
    """
    Get static information about a Shifaa dataset without downloading it.

    Args:
        dataset_type: 'mental' or 'medical'
    """
    if dataset_type.lower() in ["mental", "mental_health"]:
        return ShifaaDatasetConfig.MENTAL_HEALTH_INFO.copy()
    if dataset_type.lower() in ["medical", "medical_consultations"]:
        return ShifaaDatasetConfig.MEDICAL_CONSULTATIONS_INFO.copy()
    raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'mental' or 'medical'.")


def list_available_datasets() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of available dataset metadata."""
    return {
        "mental_health": ShifaaDatasetConfig.MENTAL_HEALTH_INFO.copy(),
        "medical_consultations": ShifaaDatasetConfig.MEDICAL_CONSULTATIONS_INFO.copy(),
    }


# Convenience aliases
load_mental_health_dataset = load_shifaa_mental_dataset
load_medical_consultations_dataset = load_shifaa_medical_dataset

