# 👁️ Shifaa Vision Module

Advanced medical image analysis using state-of-the-art deep learning models for classification and segmentation tasks.

## 📋 Table of Contents

- [Overview](#overview)
- [Available Models](#available-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Classification Models](#classification-models)
- [Segmentation Models](#segmentation-models)
- [Model Management](#model-management)
- [Advanced Usage](#advanced-usage)

---

## 🎯 Overview

The Shifaa Vision module provides pre-trained deep learning models for medical image analysis. All models are automatically downloaded from HuggingFace on first use and cached locally for subsequent runs.

**Key Features:**
- 🔄 Automatic model download from HuggingFace
- 💾 Smart caching system (download once, use forever)
- 🏥 Pre-trained on medical imaging datasets
- 📊 Support for classification and segmentation tasks
- 🎨 Built-in visualization options
- 🚀 Production-ready API

---

## 🏥 Available Models

### Classification Models

| Model | Architecture | Task | Classes | Accuracy |
|-------|--------------|------|---------|----------|
| **Brain_Tumor** | ResNet18 | Brain tumor detection | 4 | **98.55%** |
| **Chest_COVID** | ResNet50 | COVID-19 chest X-ray | 4 | **91.6%** |
| **Diabetic_Retinopathy** | EfficientNetB0 | DR severity detection | 5 | **98.55%** |
| **Eye_Disease** | EfficientNetB0 | Eye disease classification | 4 | **95%** |

### Segmentation Models

| Model | Architecture | Task | Input Size | Dice Score |
|-------|--------------|------|------------|------------|
| **CT_Heart** | U-Net | Heart CT segmentation | 224×224 | **0.9479** |
| **Skin_Cancer** | U-Net | Skin lesion segmentation | 128×128 | - |
| **Breast_Cancer** | U-Net | Breast cancer segmentation | 224×224 | **0.9179** |

---

## 📦 Installation

```bash
pip install shifaa
```

**Dependencies:**
- PyTorch
- torchvision
- Pillow
- numpy
- matplotlib

---

## 🚀 Quick Start

### Classification Example

```python
from shifaa.vision import VisionModelFactory

# Create a classification model (auto-downloads from HuggingFace)
model = VisionModelFactory.create_model(
    model_type="classification",
    model_name="Brain_Tumor"
)

# Run inference with visualization
results = model.run("path/to/brain_scan.jpg", show_image=True)

# Access results
print(f"Prediction: {results['predicted_class']}")
print(f"Confidence: {results['confidence']:.2f}%")
```

### Segmentation Example

```python
from shifaa.vision import VisionModelFactory

# Create a segmentation model
model = VisionModelFactory.create_model(
    model_type="segmentation",
    model_name="Skin_Cancer"
)

# Run segmentation with visualization
results = model.run("path/to/skin_image.jpg", show_image=True)

# Access results
image = results["image"]
mask = results["predicted_mask"]
```

**Note:** When `show_image=True`, the model displays:
- For classification: Original image with prediction and confidence
- For segmentation: Original image and predicted mask side by side

---

## 📊 Classification Models

### 1. Diabetic Retinopathy Model

**Model Information:**
- **Architecture:** EfficientNet-B0
- **Task:** Multi-class classification (5 severity levels)
- **Dataset:** [Diabetic Retinopathy Dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data)
- **Input Size:** 224×224 RGB images

**Classes:**
1. No_DR (No Diabetic Retinopathy)
2. Mild
3. Moderate
4. Severe
5. Proliferate_DR

**Performance Metrics:**
- **Accuracy:** 98.55%
- **Precision:** 0.9861
- **Recall:** 0.9855
- **F1-Score:** 0.9856

**Usage:**
```python
from shifaa.vision import VisionModelFactory

model = VisionModelFactory.create_model(
    model_type="classification",
    model_name="Diabetic_Retinopathy"
)

result = model.run("fundus_image.jpg", show_image=True)
print(f"Severity: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

**Confusion Matrix:**

![Confusion Matrix](images/DR_CM.png)

**Preprocessing:**
- Resize to 224×224
- Random horizontal flip (training)
- Random rotation ±10° (training)
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Training Details:**
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)
- **Batch Size:** 64
- **Epochs:** 30
- **Device:** CUDA/CPU

---

### 2. Eye Disease Model

**Model Information:**
- **Architecture:** EfficientNet-B0
- **Task:** Multi-class classification (4 diseases)
- **Dataset:** [Eye Disease Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- **Input Size:** 224×224 RGB images

**Classes:**
1. Cataract
2. Diabetic Retinopathy
3. Glaucoma
4. Normal

**Performance Metrics:**
- **Accuracy:** 95%
- **Precision:** 0.95
- **Recall:** 0.9555
- **F1-Score:** 0.95

**Usage:**
```python
from shifaa.vision import VisionModelFactory

model = VisionModelFactory.create_model(
    model_type="classification",
    model_name="Eye_Disease"
)

result = model.run("eye_image.jpg", show_image=True)
print(f"Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

**Confusion Matrix:**

![Confusion Matrix](images/ED_CM.png)

**Preprocessing:**
- Resize to 224×224
- Convert to tensor
- ImageNet normalization

**Training Details:**
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.0001)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)

---

### 3. Brain Tumor Model

**Model Information:**
- **Architecture:** ResNet18
- **Task:** Multi-class classification (4 tumor types)
- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Input Size:** 224×224 RGB images

**Classes:**
1. Glioma
2. Meningioma
3. Pituitary
4. No Tumor

**Performance Metrics:**
- **Accuracy:** 98.55%
- **Precision:** 0.9861
- **Recall:** 0.9855
- **F1-Score:** 0.9856

**Usage:**
```python
from shifaa.vision import VisionModelFactory

model = VisionModelFactory.create_model(
    model_type="classification",
    model_name="Brain_Tumor"
)

result = model.run("brain_mri.jpg", show_image=True)
print(f"Tumor Type: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

**Confusion Matrix:**

![Confusion Matrix](images/BT_CM.png)

**Preprocessing:**
- Resize to 224×224
- Random horizontal flip
- Random rotation ±10°
- ImageNet normalization

**Training Details:**
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 30
- **Batch Size:** 64

---

### 4. COVID-19 Chest X-ray Model

**Model Information:**
- **Architecture:** ResNet50
- **Task:** Multi-class classification (4 conditions)
- **Dataset:** [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Input Size:** 224×224 RGB images

**Classes:**
1. COVID
2. Lung_Opacity
3. Normal
4. Viral Pneumonia

**Performance Metrics:**
- **Accuracy:** 91.6%
- **Precision:** 0.92
- **Recall:** 0.91
- **F1-Score:** 0.91

**Usage:**
```python
from shifaa.vision import VisionModelFactory

model = VisionModelFactory.create_model(
    model_type="classification",
    model_name="Chest_COVID"
)

result = model.run("chest_xray.jpg", show_image=True)
print(f"Diagnosis: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

**Confusion Matrix:**

![Confusion Matrix](images/Chest_CM.png)


**Preprocessing:**
- Resize to 224×224
- Random horizontal flip
- Random rotation ±10°
- Color jitter (brightness & contrast)
- ImageNet normalization

**Training Details:**
- **Loss Function:** CrossEntropyLoss (with class weights)
- **Optimizer:** Adam (lr=0.0005)
- **Epochs:** 30
- **Batch Size:** 32

---

## 🎭 Segmentation Models

### 5. CT Heart Segmentation Model

**Model Information:**
- **Architecture:** U-Net
- **Task:** Binary segmentation of heart in CT scans
- **Dataset:** [Heart CT Dataset](https://www.kaggle.com/datasets/nikhilroxtomar/ct-heart-segmentation)
- **Input Size:** 224×224 RGB images

**Performance Metrics:**
- **Best Dice Score:** 0.9479
- **Best IoU Score:** 0.9014

**Usage:**
```python
from shifaa.vision import VisionModelFactory

model = VisionModelFactory.create_model(
    model_type="segmentation",
    model_name="CT_Heart"
)

results = model.run("heart_ct.png", show_image=True)
image = results["image"]
mask = results["predicted_mask"]
```

**Sample Results:**

![Sample Results](images/CT_H_seg.png)

**Architecture Details:**
- Encoder: 4 downsampling blocks (Conv → BatchNorm → ReLU → Dropout)
- Bottleneck: Deepest convolutional block
- Decoder: 4 upsampling blocks with skip connections
- Output: 1 channel with sigmoid activation

**Preprocessing:**
- Random horizontal flip
- Random rotation ±15°
- Random brightness & contrast adjustment
- Normalize and convert to tensor

**Training Details:**
- **Loss Function:** Combined Dice Loss + BCE Loss
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Batch Size:** 8
- **Epochs:** 100 (with early stopping)

---

### 6. Skin Cancer Segmentation Model

**Model Information:**
- **Architecture:** U-Net
- **Task:** Binary segmentation of skin lesions in dermoscopy images
- **Dataset:** Skin Lesion Mask Dataset[Skin Lesion Mask Dataset](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)
- **Input Size:** 128×128 grayscale images

**Classes:**
- Background (0)
- Lesion (1)

**Usage:**
```python
from shifaa.vision import VisionModelFactory

model = VisionModelFactory.create_model(
    model_type="segmentation",
    model_name="Skin_Cancer"
)

results = model.run("skin_lesion.jpg", show_image=True)
image = results["image"]
mask = results["predicted_mask"]
```

**Sample Results:**

![Sample Results](images/SC_seg.png)

**Architecture Details:**
- Encoder: 4 blocks (1→16→32→64→128 channels)
- Bottleneck: 256 filters
- Decoder: 4 blocks with skip connections
- Final layer: 1 output channel with sigmoid activation

**Preprocessing:**
- Images normalized to [0, 1]
- Binary masks (0=background, 1=lesion)
- Convert to tensors: shape (B, 1, H, W)

**Training Details:**
- **Loss Function:** Binary Cross-Entropy Loss
- **Optimizer:** Adam (lr=0.001)
- **Batch Size:** 16
- **Epochs:** 80 (with early stopping after 20 epochs)

**Evaluation Metrics:**
- Dice Coefficient
- Intersection over Union (IoU)

---

### 7. Breast Cancer Segmentation Model

**Model Information:**
- **Architecture:** U-Net
- **Task:** Binary segmentation of breast cancer regions
- **Dataset:** [Breast Cancer Segmentation Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Input Size:** 224×224 RGB images

**Performance Metrics:**
- **Best Dice Score:** 0.9179

**Usage:**
```python
from shifaa.vision import VisionModelFactory

model = VisionModelFactory.create_model(
    model_type="segmentation",
    model_name="Breast_Cancer"
)

results = model.run("breast_scan.png", show_image=True)
image = results["image"]
mask = results["predicted_mask"]
```

**Sample Results:**

![Sample Results](images/BC_seg2.png)

**Architecture Details:**
- Custom U-Net with 4 encoder/decoder blocks
- Skip connections for detail preservation
- Sigmoid output for binary masks

**Preprocessing:**
- Random horizontal flip
- Random rotation ±15°
- Random brightness & contrast adjustment
- Normalize and convert to tensor

**Training Details:**
- **Loss Function:** Combined Dice Loss + BCE Loss
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Batch Size:** 8
- **Epochs:** 100 (with early stopping)

**Evaluation Metrics:**
- Dice Coefficient
- IoU Score
- Validation Loss

---

## 🔧 Model Management

### List All Available Models

```python
from shifaa.vision import VisionModelFactory

models = VisionModelFactory.list_available_models()

print("Classification Models:")
for name, info in models["classification"].items():
    print(f"  • {name}: {info['architecture']}")

print("\nSegmentation Models:")
for name, info in models["segmentation"].items():
    print(f"  • {name}: {info['architecture']}")
```

### Check Model Status

```python
from shifaa.vision import VisionModelManager

manager = VisionModelManager()

# Get info about a specific model
info = manager.get_model_info("Brain_Tumor")
print(f"Downloaded: {info['is_downloaded']}")
print(f"Architecture: {info['architecture']}")
print(f"Classes: {info['classes']}")

# List all models with download status
all_models = manager.list_available_models()
for name, info in all_models.items():
    status = "✓" if info['is_downloaded'] else "✗"
    print(f"{status} {name}")
```

### Download Models Manually

```python
from shifaa.vision import download_vision_model

# Download a specific model
model_path = download_vision_model("Brain_Tumor")
print(f"Model downloaded to: {model_path}")

# Download with custom cache directory
model_path = download_vision_model(
    "Diabetic_Retinopathy",
    cache_dir="./my_models"
)
```

### Clear Model Cache

```python
from shifaa.vision import VisionModelManager

manager = VisionModelManager()

# Clear specific model
manager.clear_cache("Brain_Tumor")

# Clear all models
manager.clear_cache()
```

---

## 💡 Advanced Usage

### Custom Cache Directory

```python
from shifaa.vision import VisionModelFactory

model = VisionModelFactory.create_model(
    model_type="classification",
    model_name="Brain_Tumor",
    cache_dir="./custom_models"  # Use custom directory
)
```

### Batch Processing

```python
from pathlib import Path
from shifaa.vision import VisionModelFactory
import json

# Initialize model
model = VisionModelFactory.create_model(
    model_type="classification",
    model_name="Brain_Tumor"
)

# Process multiple images
image_dir = Path("./medical_images")
results = []

for img_path in image_dir.glob("*.jpg"):
    result = model.run(str(img_path))
    results.append({
        "image": img_path.name,
        "prediction": result['predicted_class'],
        "confidence": result['confidence']
    })

# Save results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Processed {len(results)} images")
```

### Get Model Information

```python
from shifaa.vision import VisionModelFactory

model = VisionModelFactory.create_model(
    model_type="classification",
    model_name="Brain_Tumor"
)

info = model.get_info()
print(f"Model: {info['model_name']}")
print(f"Architecture: {info['architecture']}")
print(f"Classes: {info['classes']}")
print(f"Input Size: {info['input_size']}")
```

### Disable Auto-Download

```python
from shifaa.vision import VisionModelFactory

# Raises error if model not cached
model = VisionModelFactory.create_model(
    model_type="classification",
    model_name="Brain_Tumor",
    auto_download=False  # Must be pre-downloaded
)
```

---

## 🔍 Troubleshooting

### Model Download Fails

```python
from shifaa.vision import VisionModelManager

# Force redownload
manager = VisionModelManager()
model_path = manager.get_model_path("Brain_Tumor", force=True)
```

### Out of Memory

- Process images one at a time
- Reduce batch sizes if using custom training
- Use CPU if GPU memory is limited

### Import Errors

```bash
# Install all required dependencies
pip install torch torchvision pillow numpy matplotlib
```

---

## 📝 Citation

If you use these models in your research, please cite:

```bibtex
@software{shifaa2025,
  title={Shifaa: Arabic Medical AI Platform},
  author={Ahmed Selim and Mariam Hassan and Ghada Saeed and Arwa Mohamed and Nour Ali and Hager Mohamed},
  year={2025},
  url={https://github.com/AhmedSeelim/shifaa},
  note={Datasets and models available at https://huggingface.co/Ahmed-Selem}
}
```


## 🌐 Links

- **HuggingFace Models:** [Ahmed-Selem](https://huggingface.co/Ahmed-Selem)
- **Main Documentation:** [Shifaa Package](../../README.md)
- **Colab Examples:** [Try it in Colab](https://colab.research.google.com/github/AhmedSeelim/shifaa/blob/main/Shifaa_Examples.ipynb)

---

Made with ❤️ for the medical AI community