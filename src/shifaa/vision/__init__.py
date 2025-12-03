"""
Shifaa Vision Module

Medical image analysis for classification and segmentation tasks.

Available Models:

Classification:
- Brain_Tumor: Brain tumor detection (Glioma, Meningioma, Pituitary, No Tumor)
- Chest_COVID: COVID-19 chest X-ray classification
- Diabetic_Retinopathy: Diabetic retinopathy severity detection
- Eye_Disease: Eye disease classification (Cataract, DR, Glaucoma, Normal)

Segmentation:
- CT_Heart: Heart CT scan segmentation
- Skin_Cancer: Skin lesion segmentation
- Breast_Cancer: Breast cancer segmentation

Example:
    >>> from shifaa.vision import VisionModelFactory
    >>>
    >>> # Create a classification model
    >>> model = VisionModelFactory.create_model(
    ...     model_type="classification",
    ...     model_name="Brain_Tumor"
    ... )
    >>>
    >>> # Run inference
    >>> results = model.run("brain_scan.jpg")
    >>> print(results)
"""

from .model_manager import (
    VisionModelManager,
    VisionModelConfig,
    download_vision_model,
)

from .vision_factory import (
    VisionModelFactory,
    VisionModel,
    ClassificationModel,
    SegmentationModel,
)

__all__ = [
    # Factory
    'VisionModelFactory',

    # Base Classes
    'VisionModel',
    'ClassificationModel',
    'SegmentationModel',

    # Model Management
    'VisionModelManager',
    'VisionModelConfig',
    'download_vision_model',
]

__version__ = '0.1.0'