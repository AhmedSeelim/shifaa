"""
Vision Model Manager

Handles downloading and managing Shifaa vision models from HuggingFace.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionModelConfig:
    """Configuration for vision models on HuggingFace."""

    # Classification Models
    DIABETIC_RETINOPATHY = {
        "repo_id": "Ahmed-Selem/Shifaa-Diabetic-Retinopathy-EfficientNetB0",
        "filename": "DiabeticRetinopathy.pth",
        "type": "classification",
        "architecture": "EfficientNetB0",
        "classes": ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"],
        "input_size": (224, 224)
    }

    EYE_DISEASE = {
        "repo_id": "Ahmed-Selem/Shifaa-Eye-Disease-EfficientNetB0",
        "filename": "efficientnet_b0_Eye_Diseases.pth",
        "type": "classification",
        "architecture": "EfficientNetB0",
        "classes": ["Cataract", "Diabetic_Retinopathy", "Glaucoma", "Normal"],
        "input_size": (224, 224)
    }

    BRAIN_TUMOR = {
        "repo_id": "Ahmed-Selem/Shifaa-Brain-Tumor-ResNet18",
        "filename": "braintumor_model_weights.pth",
        "type": "classification",
        "architecture": "ResNet18",
        "classes": ["Glioma", "Meningioma", "Pituitary", "No_Tumor"],
        "input_size": (224, 224)
    }

    COVID_CHEST_XRAY = {
        "repo_id": "Ahmed-Selem/Shifaa-COVID-Chest-Xray-ResNet50",
        "filename": "covid.pth",
        "type": "classification",
        "architecture": "ResNet50",
        "classes": ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"],
        "input_size": (224, 224)
    }

    # Segmentation Models
    HEART_CT_SEGMENTATION = {
        "repo_id": "Ahmed-Selem/Shifaa-Heart-CT-UNet",
        "filename": "model (6).pth",
        "type": "segmentation",
        "architecture": "UNet",
        "task": "binary_segmentation",
        "input_size": (224, 224)
    }

    SKIN_CANCER_SEGMENTATION = {
        "repo_id": "Ahmed-Selem/Shifaa-Skin-Cancer-UNet-Segmentation",
        "filename": "Shifaa-Skin-Cancer-UNet-Segmentation.pth",
        "type": "segmentation",
        "architecture": "UNet",
        "task": "binary_segmentation",
        "input_size": (128, 128)
    }

    BREAST_CANCER_SEGMENTATION = {
        "repo_id": "Ahmed-Selem/Shifaa-Breast-Cancer-UNet-Segmentation",
        "filename": "best_unet_model7995.pth",
        "type": "segmentation",
        "architecture": "UNet",
        "task": "binary_segmentation",
        "input_size": (224, 224)
    }

    # Model registry
    MODELS = {
        "Diabetic_Retinopathy": DIABETIC_RETINOPATHY,
        "Eye_Disease": EYE_DISEASE,
        "Brain_Tumor": BRAIN_TUMOR,
        "Chest_COVID": COVID_CHEST_XRAY,
        "CT_Heart": HEART_CT_SEGMENTATION,
        "Skin_Cancer": SKIN_CANCER_SEGMENTATION,
        "Breast_Cancer": BREAST_CANCER_SEGMENTATION,
    }

    DEFAULT_CACHE_DIR = Path.home() / ".shifaa" / "vision_models"


class VisionModelManager:
    """Manages vision model downloads and caching."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the Vision Model Manager.

        Args:
            cache_dir: Custom cache directory for models.
                      Defaults to ~/.shifaa/vision_models
        """
        self.cache_dir = cache_dir or VisionModelConfig.DEFAULT_CACHE_DIR
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_model_downloaded(self, model_name: str) -> bool:
        """
        Check if a model is already downloaded.

        Args:
            model_name: Name of the model

        Returns:
            True if model exists locally
        """
        if model_name not in VisionModelConfig.MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        config = VisionModelConfig.MODELS[model_name]
        model_path = self.cache_dir / config["filename"]

        return model_path.exists() and model_path.stat().st_size > 0

    def get_model_path(
            self,
            model_name: str,
            auto_download: bool = True,
            force: bool = False
    ) -> Path:
        """
        Get the path to a model, downloading if necessary.

        Args:
            model_name: Name of the model
            auto_download: Automatically download if not present
            force: Force redownload even if exists

        Returns:
            Path to the model file

        Raises:
            ValueError: If model name is unknown
            FileNotFoundError: If model not found and auto_download is False
        """
        if model_name not in VisionModelConfig.MODELS:
            available = ", ".join(VisionModelConfig.MODELS.keys())
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {available}"
            )

        config = VisionModelConfig.MODELS[model_name]
        model_path = self.cache_dir / config["filename"]

        # Check if already downloaded
        if self.is_model_downloaded(model_name) and not force:
            logger.info(f"✓ Model '{model_name}' already cached at: {model_path}")
            return model_path

        # Download if needed
        if auto_download or force:
            return self._download_model(model_name, force=force)
        else:
            raise FileNotFoundError(
                f"Model '{model_name}' not found at {model_path}. "
                "Set auto_download=True to download automatically."
            )

    def _download_model(self, model_name: str, force: bool = False) -> Path:
        """
        Download a model from HuggingFace.

        Args:
            model_name: Name of the model
            force: Force redownload

        Returns:
            Path to the downloaded model
        """
        config = VisionModelConfig.MODELS[model_name]

        logger.info(f"Downloading {model_name} from HuggingFace...")
        logger.info(f"Repository: {config['repo_id']}")

        try:
            # Download from HuggingFace
            downloaded_path = hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename"],
                cache_dir=self.cache_dir,
                force_download=force
            )

            # Move to our cache directory with expected filename
            target_path = self.cache_dir / config["filename"]

            if Path(downloaded_path) != target_path:
                import shutil
                shutil.copy2(downloaded_path, target_path)

            logger.info(f"✓ Successfully downloaded {model_name}")
            logger.info(f"✓ Cached at: {target_path}")

            return target_path

        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            raise RuntimeError(f"Model download failed: {str(e)}")

    def download_all_models(self, force: bool = False):
        """
        Download all available models.

        Args:
            force: Force redownload even if exists
        """
        logger.info("Downloading all Shifaa vision models...")

        for model_name in VisionModelConfig.MODELS.keys():
            try:
                self.get_model_path(model_name, force=force)
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {str(e)}")

        logger.info("✓ All models downloaded")

    def get_model_info(self, model_name: str) -> Dict:
        """
        Get information about a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information
        """
        if model_name not in VisionModelConfig.MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        config = VisionModelConfig.MODELS[model_name].copy()
        config["is_downloaded"] = self.is_model_downloaded(model_name)

        if config["is_downloaded"]:
            model_path = self.cache_dir / config["filename"]
            config["path"] = str(model_path)
            config["size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2)

        return config

    def list_available_models(self) -> Dict[str, Dict]:
        """
        List all available models with their status.

        Returns:
            Dictionary mapping model names to their info
        """
        return {
            name: self.get_model_info(name)
            for name in VisionModelConfig.MODELS.keys()
        }

    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear model cache.

        Args:
            model_name: Specific model to clear, or None for all models
        """
        if model_name:
            if model_name not in VisionModelConfig.MODELS:
                raise ValueError(f"Unknown model: {model_name}")

            config = VisionModelConfig.MODELS[model_name]
            model_path = self.cache_dir / config["filename"]

            if model_path.exists():
                model_path.unlink()
                logger.info(f"✓ Cleared cache for {model_name}")
            else:
                logger.info(f"No cache to clear for {model_name}")
        else:
            # Clear all models
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("✓ Cleared all model caches")


def download_vision_model(
        model_name: str,
        cache_dir: Optional[str] = None,
        force: bool = False
) -> str:
    """
    Convenience function to download a vision model.

    Args:
        model_name: Name of the model to download
        cache_dir: Custom cache directory
        force: Force redownload

    Returns:
        Path to the downloaded model as string

    Example:
        >>> from shifaa.vision import download_vision_model
        >>> model_path = download_vision_model("Brain_Tumor")
        >>> print(f"Model at: {model_path}")
    """
    manager = VisionModelManager(
        cache_dir=Path(cache_dir) if cache_dir else None
    )
    return str(manager.get_model_path(model_name, force=force))