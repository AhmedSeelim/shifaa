"""
Shifaa Datasets Usage Examples

This module demonstrates practical usage of the Shifaa dataset loader API.
Run this file directly to see all examples in action.
"""

from shifaa.datasets import (
    load_shifaa_mental_dataset,
    load_shifaa_medical_dataset,
    get_dataset_info,
    list_available_datasets
)


def example_1_basic_loading():
    """Example 1: Basic dataset loading"""
    print("=" * 60)
    print("Example 1: Basic Dataset Loading")
    print("=" * 60)

    print("\nLoading mental health dataset...")
    mental_data = load_shifaa_mental_dataset(split="train")
    print(f"✓ Loaded {len(mental_data)} mental health consultations")

    print("\nLoading medical consultations dataset...")
    medical_data = load_shifaa_medical_dataset(split="train")
    print(f"✓ Loaded {len(medical_data)} medical consultations")


def example_2_dataset_info():
    """Example 2: Getting dataset information"""
    print("\n" + "=" * 60)
    print("Example 2: Dataset Information")
    print("=" * 60)

    print("\nAvailable Datasets:")
    datasets = list_available_datasets()

    for name, info in datasets.items():
        print(f"\n{name.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    print("\nMental Health Dataset Info:")
    print(get_dataset_info("mental"))


def example_3_preview_samples():
    """Example 3: Preview first samples"""
    print("\n" + "=" * 60)
    print("Example 3: Dataset Preview")
    print("=" * 60)

    medical_data = load_shifaa_medical_dataset(split="train")

    print("\nFirst 3 consultations:\n")
    for i in range(3):
        sample = medical_data[i]
        print(f"--- Consultation {i+1} ---")
        print(f"Question: {sample.get('Question')}")
        print(f"Answer: {sample.get('Answer')[:200]}...")
        print()


def example_4_custom_cache():
    """Example 4: Using a custom cache directory"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Cache")
    print("=" * 60)

    dataset = load_shifaa_medical_dataset(cache_dir="./shifaa_cache")
    print("✓ Dataset cached in ./shifaa_cache")
    print(f"✓ Loaded {len(dataset)} consultations")


def example_5_to_pandas():
    """Example 5: Convert dataset to pandas DataFrame"""
    print("\n" + "=" * 60)
    print("Example 5: Pandas Conversion")
    print("=" * 60)

    try:
        import pandas as pd

        medical_data = load_shifaa_medical_dataset(split="train")

        print("\nConverting to pandas DataFrame...")
        df = medical_data.to_pandas()

        print(f"✓ DataFrame shape: {df.shape}")
        print(df.head())

    except ImportError:
        print("\n⚠ pandas not installed. Install with: pip install pandas")


def main():
    print("\n🏥 Shifaa Datasets - Usage Examples")
    print("=" * 60)

    examples = [
        example_1_basic_loading,
        example_2_dataset_info,
        example_3_preview_samples,
        example_4_custom_cache,
        example_5_to_pandas,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n⚠ Error in {example.__name__}: {e}")

    print("\n✅ All examples executed successfully.")


if __name__ == "__main__":
    main()
