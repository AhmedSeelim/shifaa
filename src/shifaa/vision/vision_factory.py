"""
Vision Model Factory

Factory pattern for creating and managing vision models with automatic model downloading.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

from .model_manager import VisionModelManager, VisionModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionModel(ABC):
    """Abstract base class for all vision models."""

    @abstractmethod
    def run(self, image_path: str, **kwargs) -> Any:
        """
        Abstract run method to be implemented by all subclasses.

        Args:
            image_path: Path to the input image
            **kwargs: Additional arguments

        Returns:
            Model-specific output
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass


class ClassificationModel(VisionModel):
    """Class for running classification models."""

    def __init__(
            self,
            model_name: str,
            auto_download: bool = True,
            cache_dir: Optional[str] = None
    ):
        """
        Initialize a classification model.

        Args:
            model_name: Name of the classification model
            auto_download: Automatically download model if not present
            cache_dir: Custom cache directory for models
        """
        self.model_name = model_name

        # Validate model name
        if model_name not in VisionModelConfig.MODELS:
            available = [k for k, v in VisionModelConfig.MODELS.items()
                         if v["type"] == "classification"]
            raise ValueError(
                f"Unknown classification model: {model_name}. "
                f"Available: {', '.join(available)}"
            )

        # Check if it's a classification model
        model_config = VisionModelConfig.MODELS[model_name]
        if model_config["type"] != "classification":
            raise ValueError(
                f"{model_name} is not a classification model. "
                f"It's a {model_config['type']} model."
            )

        # Get model path (downloads if needed)
        self.model_manager = VisionModelManager(cache_dir=cache_dir)
        self.model_path = self.model_manager.get_model_path(
            model_name,
            auto_download=auto_download
        )

        self.config = model_config
        self.model = self._load_model()

    def _load_model(self) -> Any:
        """Load the appropriate model based on model_name."""
        try:
            if self.model_name == "Brain_Tumor":
                from .classification.BrainTumor import BrainTumorClassifier
                return BrainTumorClassifier.BrainTumorClassifier(
                    model_weights_path=str(self.model_path)
                )

            elif self.model_name == "Chest_COVID":
                from .classification.covid import COVIDClassifier
                return COVIDClassifier.COVIDClassifier(str(self.model_path))

            elif self.model_name == "Diabetic_Retinopathy":
                from .classification.DR import DiabeticRetinopathyClassifier
                return DiabeticRetinopathyClassifier.DiabeticRetinopathyClassifier(
                    model_path=str(self.model_path)
                )

            elif self.model_name == "Eye_Disease":
                from .classification.Eye_Disease import EyeDiseaseClassifier
                return EyeDiseaseClassifier.EyeDiseaseClassifier(
                    model_path=str(self.model_path)
                )

            else:
                raise ValueError(f"Unknown classification model: {self.model_name}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import classifier for {self.model_name}. "
                f"Make sure all dependencies are installed. Error: {str(e)}"
            )

    def run(
            self,
            image_path: str,
            show_image: bool = False,
            **kwargs
    ) -> Dict:
        """
        Run classification on the given image.

        Args:
            image_path: Path to the input image
            show_image: Whether to display the image (default: False)
            **kwargs: Additional arguments

        Returns:
            Dictionary with classification results
        """
        try:
            if self.model_name in ["Brain_Tumor", "Diabetic_Retinopathy", "Eye_Disease","Chest_COVID"]:
                return self.model.infer(image_path, show_image=show_image)


            else:
                raise ValueError(f"No implementation for model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error running {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        """Get information about the classification model."""
        return {
            "model_name": self.model_name,
            "type": "classification",
            "architecture": self.config["architecture"],
            "classes": self.config["classes"],
            "num_classes": len(self.config["classes"]),
            "input_size": self.config["input_size"],
            "model_path": str(self.model_path),
            "is_loaded": self.model is not None
        }


class SegmentationModel(VisionModel):
    """Class for running segmentation models."""

    def __init__(
            self,
            model_name: str,
            auto_download: bool = True,
            cache_dir: Optional[str] = None
    ):
        """
        Initialize a segmentation model.

        Args:
            model_name: Name of the segmentation model
            auto_download: Automatically download model if not present
            cache_dir: Custom cache directory for models
        """
        self.model_name = model_name

        # Validate model name
        if model_name not in VisionModelConfig.MODELS:
            available = [k for k, v in VisionModelConfig.MODELS.items()
                         if v["type"] == "segmentation"]
            raise ValueError(
                f"Unknown segmentation model: {model_name}. "
                f"Available: {', '.join(available)}"
            )

        # Check if it's a segmentation model
        model_config = VisionModelConfig.MODELS[model_name]
        if model_config["type"] != "segmentation":
            raise ValueError(
                f"{model_name} is not a segmentation model. "
                f"It's a {model_config['type']} model."
            )

        # Get model path (downloads if needed)
        self.model_manager = VisionModelManager(cache_dir=cache_dir)
        self.model_path = self.model_manager.get_model_path(
            model_name,
            auto_download=auto_download
        )

        self.config = model_config
        self.model = self._load_model()

    def _load_model(self):
        """Load the appropriate segmentation model."""
        input_size = self.config.get("input_size", (128, 128))

        try:
            if self.model_name == "CT_Heart":
                from .segmentation.CT_Heart.CT_Heart import CTHeart
                return CTHeart(str(self.model_path))

            elif self.model_name == "Skin_Cancer":
                from .segmentation.Skin_Cancer import skin_cancer
                return skin_cancer.skin_cancer(str(self.model_path), input_size=input_size)

            elif self.model_name == "Breast_Cancer":
                from .segmentation.Breast_Cancer import breast_cancer
                return breast_cancer.breast_cancer(str(self.model_path), input_size=input_size)

            else:
                raise ValueError(f"Unknown segmentation model: {self.model_name}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import segmentation model for {self.model_name}. "
                f"Make sure all dependencies are installed. Error: {str(e)}"
            )

    def run(
            self,
            image_path: str,
            show_image: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Run segmentation on the given image.

        Args:
            image_path: Path to the input image
            show: Whether to display results (default: False)
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'image' and 'predicted_mask' keys
        """
        try:
            result = self.model.predict(image_path, show_image=show_image)
            return {
                'image': result['image'],
                'predicted_mask': result['predicted_mask']
            }
        except Exception as e:
            logger.error(f"Error running {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        """Get information about the segmentation model."""
        return {
            "model_name": self.model_name,
            "type": "segmentation",
            "architecture": self.config["architecture"],
            "task": self.config.get("task", "segmentation"),
            "input_size": self.config["input_size"],
            "model_path": str(self.model_path),
            "is_loaded": self.model is not None
        }


class VisionModelFactory:
    """
    Factory class for creating appropriate vision model instances.

    This factory handles model creation with automatic downloading from HuggingFace.
    """

    @staticmethod
    def create_model(
            model_type: str,
            model_name: str,
            auto_download: bool = True,
            cache_dir: Optional[str] = None,
            **kwargs
    ) -> VisionModel:
        """
        Create and return an appropriate model instance based on model_type.

        Args:
            model_type: Type of vision model ('classification', 'segmentation')
            model_name: Specific model name
            auto_download: Automatically download model if not present
            cache_dir: Custom cache directory for models
            **kwargs: Additional arguments for model initialization

        Returns:
            An instance of the appropriate VisionModel subclass

        Raises:
            ValueError: If model_type is unknown

        Example:
            >>> from shifaa.vision import VisionModelFactory
            >>> model = VisionModelFactory.create_model(
            ...     model_type="classification",
            ...     model_name="Brain_Tumor"
            ... )
            >>> results = model.run("brain_scan.jpg")
        """
        model_type_lower = model_type.lower()

        if model_type_lower == "classification":
            return ClassificationModel(
                model_name=model_name,
                auto_download=auto_download,
                cache_dir=cache_dir
            )

        elif model_type_lower == "segmentation":
            return SegmentationModel(
                model_name=model_name,
                auto_download=auto_download,
                cache_dir=cache_dir
            )

        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                "Available types: 'classification', 'segmentation'"
            )

    @staticmethod
    def list_available_models() -> Dict[str, Dict]:
        """
        List all available models grouped by type.

        Returns:
            Dictionary with model types as keys and lists of models as values
        """
        manager = VisionModelManager()
        all_models = manager.list_available_models()

        classification_models = {
            name: info for name, info in all_models.items()
            if info["type"] == "classification"
        }

        segmentation_models = {
            name: info for name, info in all_models.items()
            if info["type"] == "segmentation"
        }

        return {
            "classification": classification_models,
            "segmentation": segmentation_models
        }