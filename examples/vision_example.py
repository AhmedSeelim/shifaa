"""
Shifaa Vision Module Usage Examples

Demonstrates how to use the Vision module for medical image analysis.
"""

from pathlib import Path
from shifaa.vision import (
    VisionModelFactory,
    VisionModelManager,
    download_vision_model
)


def example_1_list_models():
    """Example 1: List all available models"""
    print("=" * 60)
    print("Example 1: List Available Models")
    print("=" * 60)

    models = VisionModelFactory.list_available_models()

    print("\n### Classification Models ###")
    for name, info in models["classification"].items():
        status = "✓" if info.get('is_downloaded', False) else "✗"
        print(f"\n{status} {name}")
        print(f"   Architecture: {info['architecture']}")
        print(f"   Classes: {len(info['classes'])}")
        print(f"   Input Size: {info['input_size']}")

    print("\n### Segmentation Models ###")
    for name, info in models["segmentation"].items():
        status = "✓" if info.get('is_downloaded', False) else "✗"
        print(f"\n{status} {name}")
        print(f"   Architecture: {info['architecture']}")
        print(f"   Task: {info.get('task', 'segmentation')}")
        print(f"   Input Size: {info['input_size']}")


def example_2_classification_basic():
    """Example 2: Basic classification"""
    print("\n" + "=" * 60)
    print("Example 2: Brain Tumor Classification")
    print("=" * 60)

    try:
        # Create model (auto-downloads if not cached)
        print("\nInitializing Brain Tumor classifier...")
        model = VisionModelFactory.create_model(
            model_type="classification",
            model_name="Brain_Tumor"
        )

        print("✓ Model loaded successfully")

        # Get model info
        info = model.get_info()
        print(f"\nModel Information:")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Classes: {', '.join(info['classes'])}")
        print(f"  Input Size: {info['input_size']}")

        # NOTE: Actual inference requires an image file
        print("\nTo run inference:")
        print("  result = model.run('path/to/brain_scan.jpg')")
        print("  print(result['predicted_class'])")

    except Exception as e:
        print(f"⚠ Error: {str(e)}")
        print("This is expected if model files aren't available yet.")


def example_3_segmentation_basic():
    """Example 3: Basic segmentation"""
    print("\n" + "=" * 60)
    print("Example 3: Heart CT Segmentation")
    print("=" * 60)

    try:
        # Create segmentation model
        print("\nInitializing Heart CT segmentation model...")
        model = VisionModelFactory.create_model(
            model_type="segmentation",
            model_name="CT_Heart"
        )

        print("✓ Model loaded successfully")

        # Get model info
        info = model.get_info()
        print(f"\nModel Information:")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Task: {info['task']}")
        print(f"  Input Size: {info['input_size']}")

        # NOTE: Actual inference requires an image file
        print("\nTo run inference:")
        print("  image, mask = model.run('path/to/heart_ct.png')")
        print("  # Visualize with matplotlib")

    except Exception as e:
        print(f"⚠ Error: {str(e)}")


def example_4_model_management():
    """Example 4: Model management"""
    print("\n" + "=" * 60)
    print("Example 4: Model Management")
    print("=" * 60)

    manager = VisionModelManager()

    # Check which models are downloaded
    print("\nModel Download Status:")
    all_models = manager.list_available_models()

    for name, info in all_models.items():
        status = "✓ Downloaded" if info['is_downloaded'] else "✗ Not Downloaded"
        print(f"  {status}: {name}")

        if info['is_downloaded']:
            print(f"    Size: {info.get('size_mb', 'N/A')} MB")
            print(f"    Path: {info.get('path', 'N/A')}")


def example_5_manual_download():
    """Example 5: Manual model download"""
    print("\n" + "=" * 60)
    print("Example 5: Manual Model Download")
    print("=" * 60)

    # Download a specific model
    print("\nDownloading Diabetic Retinopathy model...")
    print("(This will download ~100MB, may take a few minutes)")

    try:
        model_path = download_vision_model(
            "Diabetic_Retinopathy",
            cache_dir=None  # Use default cache
        )
        print(f"✓ Model downloaded to: {model_path}")

    except Exception as e:
        print(f"⚠ Download failed: {str(e)}")


def example_6_custom_cache():
    """Example 6: Using custom cache directory"""
    print("\n" + "=" * 60)
    print("Example 6: Custom Cache Directory")
    print("=" * 60)

    custom_cache = "./my_vision_models"
    print(f"\nUsing custom cache: {custom_cache}")

    try:
        model = VisionModelFactory.create_model(
            model_type="classification",
            model_name="Eye_Disease",
            cache_dir=custom_cache
        )

        print("✓ Model initialized with custom cache")
        print(f"  Cache location: {custom_cache}")

    except Exception as e:
        print(f"⚠ Error: {str(e)}")


def example_7_batch_processing_demo():
    """Example 7: Batch processing demonstration"""
    print("\n" + "=" * 60)
    print("Example 7: Batch Processing (Demo)")
    print("=" * 60)

    print("\nExample code for batch processing:")
    print("""
from pathlib import Path
from shifaa.vision import VisionModelFactory

# Initialize model
model = VisionModelFactory.create_model(
    model_type="classification",
    model_name="Brain_Tumor"
)

# Process directory of images
image_dir = Path("./medical_images")
results = []

for img_path in image_dir.glob("*.jpg"):
    result = model.run(str(img_path))
    results.append({
        "image": img_path.name,
        "prediction": result['label'],
        "confidence": result['confidence']
    })

# Save results
import json
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
    """)


def example_8_all_classification_models():
    """Example 8: Load all classification models"""
    print("\n" + "=" * 60)
    print("Example 8: All Classification Models")
    print("=" * 60)

    classification_models = [
        "Brain_Tumor",
        "Chest_COVID",
        "Diabetic_Retinopathy",
        "Eye_Disease"
    ]

    print("\nAttempting to initialize all classification models...")
    print("(This will download models if not cached)")

    for model_name in classification_models:
        print(f"\n{model_name}:")
        try:
            model = VisionModelFactory.create_model(
                model_type="classification",
                model_name=model_name
            )
            info = model.get_info()
            print(f"  ✓ Loaded: {info['architecture']}")
            print(f"  Classes: {', '.join(info['classes'])}")
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")


def example_9_all_segmentation_models():
    """Example 9: Load all segmentation models"""
    print("\n" + "=" * 60)
    print("Example 9: All Segmentation Models")
    print("=" * 60)

    segmentation_models = [
        "CT_Heart",
        "Skin_Cancer",
        "Breast_Cancer"
    ]

    print("\nAttempting to initialize all segmentation models...")

    for model_name in segmentation_models:
        print(f"\n{model_name}:")
        try:
            model = VisionModelFactory.create_model(
                model_type="segmentation",
                model_name=model_name
            )
            info = model.get_info()
            print(f"  ✓ Loaded: {info['architecture']}")
            print(f"  Task: {info['task']}")
            print(f"  Input Size: {info['input_size']}")
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")


def main():
    """Run all examples"""
    print("\n🏥 Shifaa Vision Module Examples")
    print("=" * 60)
    print("\nNote: Some examples require actual image files to run fully.")
    print("Models will be downloaded from HuggingFace on first use.\n")

    examples = [
        example_1_list_models,
        example_2_classification_basic,
        example_3_segmentation_basic,
        example_4_model_management,
        example_5_manual_download,
        example_6_custom_cache,
        example_7_batch_processing_demo,
        # example_8_all_classification_models,  # Uncomment to download all
        # example_9_all_segmentation_models,     # Uncomment to download all
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n⚠ Error in {example.__name__}: {str(e)}")

    print("\n" + "=" * 60)
    print("✓ Examples complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. Prepare medical images for analysis")
    print("  2. Run inference with actual image files")
    print("  3. Visualize results")
    print("  4. Integrate into your medical AI pipeline")


if __name__ == "__main__":
    main()