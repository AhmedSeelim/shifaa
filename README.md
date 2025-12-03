# 🏥 Shifaa - Arabic Medical AI 

<div align="center">

<!-- Logo Placeholder -->
<img src="Shifaa_Logo.png" alt="Shifaa Logo"/>

**Revolutionizing Arabic Medical AI for the MENA Region**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/Ahmed-Selem)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AhmedSeelim/shifaa/blob/main/Shifaa_Examples.ipynb)

[Features](#-key-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) 

</div>

---

## 🌟 What is Shifaa?

**Shifaa** is a comprehensive medical AI specifically designed for Arabic-speaking healthcare professionals, researchers, and developers across the MENA region.

Shifaa addresses the critical gap in Arabic healthcare technology by providing:
- **Curated Arabic Medical Datasets** - Access to 120K+ real medical consultations
- **Intelligent RAG System** - Medical information retrieval with 84K+ knowledge base
- **Pre-trained Vision Models** - 7 medical imaging models for diagnosis and segmentation

---

## ✨ Key Features

### 📊 **Comprehensive Datasets**
- **35,648** Arabic mental health consultations across 7 specializations
- **84,422** medical consultations covering 16 specializations and 585 diagnoses
- High-quality, structured data with no missing values
- Easy access through HuggingFace integration

### 🤖 **Intelligent RAG System**
- Automatic medical specialty detection from queries
- Semantic search over 84,000+ consultations
- Hierarchical topic path identification (585 medical topics)
- Multi-stage pipeline: Query → Specialty → Topics → Retrieval → Insights
- Supports Arabic and multilingual queries

### 👁️ **Medical Vision Models**
- **4 Classification Models:** Brain tumors, COVID-19, diabetic retinopathy, eye diseases
- **3 Segmentation Models:** Heart CT, skin cancer, breast cancer
- Accuracy up to **98.55%** on medical imaging tasks
- Automatic model download and caching from HuggingFace
- Built-in visualization and inference tools

---

## 🚀 Installation

### Quick Install

```bash
pip install shifaa
```

### From Source

```bash
git clone https://github.com/AhmedSeelim/shifaa.git
cd shifaa
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- LangChain
- ChromaDB

For complete requirements, see [requirements.txt](requirements.txt).

---

## ⚡ Quick Start

### 1. Load Datasets

```python
from shifaa.datasets import load_shifaa_mental_dataset, load_shifaa_medical_dataset

# Load mental health consultations
mental_data = load_shifaa_mental_dataset()
print(f"Loaded {len(mental_data)} mental health consultations")

# Load medical consultations
medical_data = load_shifaa_medical_dataset()
print(f"Loaded {len(medical_data)} medical consultations")
```

### 2. Use Medical RAG

```python
from shifaa.rag import MedicalRAGSystem
import os

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

# Initialize RAG system (auto-downloads vector database)
rag = MedicalRAGSystem()

# Query in Arabic
results = rag.process_query("ما هي أعراض السكري؟")

# Access results
print(f"Specialties: {[s.specialty for s in results.specialties]}")
print(f"Insights: {[i.information for i in results.insights]}")
```

### 3. Analyze Medical Images

```python
from shifaa.vision import VisionModelFactory

# Classification: Brain tumor detection
model = VisionModelFactory.create_model(
    model_type="classification",
    model_name="Brain_Tumor"
)
result = model.run("brain_scan.jpg", show_image=True)
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}%")

# Segmentation: Skin cancer detection
seg_model = VisionModelFactory.create_model(
    model_type="segmentation",
    model_name="Skin_Cancer"
)
results = seg_model.run("skin_lesion.jpg", show_image=True)
image = results["image"]
mask = results["predicted_mask"]
```

---

## 📚 Modules Overview

### 📊 Datasets Module

Access curated Arabic medical datasets hosted on HuggingFace.

**Features:**
- Simple API for loading datasets
- Automatic caching
- 120K+ consultations covering mental health and general medicine

**Example:**
```python
from shifaa.datasets import load_shifaa_medical_dataset
dataset = load_shifaa_medical_dataset()
```

**Available Datasets:**
- [Mental Health Consultations](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations) - 35,648 consultations
- [Medical Consultations](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Medical_Consultations) - 84,422 consultations

[→ Full Datasets Documentation](https://github.com/AhmedSeelim/shifaa/blob/main/src/shifaa/datasets/README.md)

---

### 🤖 RAG Module

Medical Retrieval-Augmented Generation system for intelligent information retrieval.

**Features:**
- 4-stage pipeline: Specialty detection → Topic identification → Consultation retrieval → Insight extraction
- Semantic search over 84K+ consultations
- 585 hierarchical medical topics
- Arabic language support

**Example:**
```python
from shifaa.rag import MedicalRAGSystem

rag = MedicalRAGSystem()
results = rag.process_query("كيف أعالج الصداع المزمن؟")
```

**Pipeline Architecture:**
```
User Query
    ↓
Specialty Detection (23 specialties)
    ↓
Topic Path Identification (585 topics)
    ↓
Semantic Retrieval (84K+ consultations)
    ↓
Medical Insights Extraction
```

[→ Full RAG Documentation](https://github.com/AhmedSeelim/shifaa/blob/main/src/shifaa/rag/README.md)

---

### 👁️ Vision Module

Pre-trained deep learning models for medical image analysis.

**Classification Models (4):**

| Model | Task | Accuracy | HuggingFace |
|-------|------|----------|-------------|
| Brain Tumor | Tumor classification | 98.55% | [Link](https://huggingface.co/Ahmed-Selem/Shifaa-Brain-Tumor-ResNet18) |
| COVID-19 | Chest X-ray diagnosis | 91.6% | [Link](https://huggingface.co/Ahmed-Selem/Shifaa-COVID-Chest-Xray-ResNet50) |
| Diabetic Retinopathy | DR severity detection | 98.55% | [Link](https://huggingface.co/Ahmed-Selem/Shifaa-Diabetic-Retinopathy-EfficientNetB0) |
| Eye Disease | Eye disease classification | 95% | [Link](https://huggingface.co/Ahmed-Selem/Shifaa-Eye-Disease-EfficientNetB0) |

**Segmentation Models (3):**

| Model | Task | Dice Score | HuggingFace |
|-------|------|-----------|-------------|
| Heart CT | Heart segmentation | 0.9479 | [Link](https://huggingface.co/Ahmed-Selem/Shifaa-Heart-CT-UNet) |
| Skin Cancer | Lesion segmentation | 0.9175 | [Link](https://huggingface.co/Ahmed-Selem/Shifaa-Skin-Cancer-UNet-Segmentation) |
| Breast Cancer | Tumor segmentation | 0.9179 | [Link](https://huggingface.co/Ahmed-Selem/Shifaa-Breast-Cancer-UNet-Segmentation) |

**Example:**
```python
from shifaa.vision import VisionModelFactory

model = VisionModelFactory.create_model("classification", "Brain_Tumor")
result = model.run("brain_scan.jpg", show_image=True)
```

[→ Full Vision Documentation](https://github.com/AhmedSeelim/shifaa/blob/main/src/shifaa/vision/README.md)

---

## 📖 Documentation

- **[Installation Guide](INSTALLATION.md)** - Detailed setup instructions
- **[Datasets Documentation](shifaa/datasets/README.md)** - Dataset details and usage
- **[RAG Documentation](shifaa/rag/README.md)** - RAG system guide
- **[Vision Documentation](shifaa/vision/README.md)** - Medical imaging models
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Colab Notebook](https://colab.research.google.com/github/AhmedSeelim/shifaa/blob/main/Shifaa_Examples.ipynb)** - Interactive examples

---

## 🎓 Examples

### Try in Google Colab

The easiest way to get started is with our interactive Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AhmedSeelim/shifaa/blob/main/Shifaa_Examples.ipynb)

### Local Examples

```bash
# Clone the repository
git clone https://github.com/AhmedSeelim/shifaa.git
cd shifaa

# Run examples
python examples/datasets_example.py
python examples/rag_example.py
python examples/vision_example.py
```

---

## 📊 Datasets on HuggingFace

All Shifaa datasets are hosted on HuggingFace for easy access:

- [**Mental Health Consultations**](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations)
  - 35,648 consultations
  - 7 specializations
  - 123 specific diagnoses

- [**Medical Consultations**](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Medical_Consultations)
  - 84,422 consultations
  - 16 specializations
  - 585 hierarchical diagnoses
  - Average answer length: 273 words

---

## 🤖 Models on HuggingFace

All vision models are available on HuggingFace:

**Classification:**
- [Brain Tumor - ResNet18](https://huggingface.co/Ahmed-Selem/Shifaa-Brain-Tumor-ResNet18)
- [COVID-19 - ResNet50](https://huggingface.co/Ahmed-Selem/Shifaa-COVID-Chest-Xray-ResNet50)
- [Diabetic Retinopathy - EfficientNetB0](https://huggingface.co/Ahmed-Selem/Shifaa-Diabetic-Retinopathy-EfficientNetB0)
- [Eye Disease - EfficientNetB0](https://huggingface.co/Ahmed-Selem/Shifaa-Eye-Disease-EfficientNetB0)

**Segmentation:**
- [Heart CT - U-Net](https://huggingface.co/Ahmed-Selem/Shifaa-Heart-CT-UNet)
- [Skin Cancer - U-Net](https://huggingface.co/Ahmed-Selem/Shifaa-Skin-Cancer-UNet-Segmentation)
- [Breast Cancer - U-Net](https://huggingface.co/Ahmed-Selem/Shifaa-Breast-Cancer-UNet-Segmentation)

**RAG Vector Database:**
- [Medical RAG Vector DB](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Medical_RAG_VectorDB)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Shifaa Ecosystem                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Datasets   │  │     RAG      │  │    Vision    │      │
│  │              │  │              │  │              │      │
│  │ • Mental     │  │ • Medical    │  │ • Brain      │      │
│  │   Health     │  │   RAG        │  │   Tumor      │      │
│  │   (35K)      │  │ • Vector DB  │  │ • COVID-19   │      │
│  │ • Medical    │  │   (84K)      │  │ • Diabetic   │      │
│  │   (84K)      │  │ • Semantic   │  │   Retinopathy│      │
│  │              │  │   Search     │  │ • Segmentation│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  HuggingFace Integration • Automatic Caching • Easy API     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤝 Contributing

We welcome contributions from the community! Whether it's:

- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🌐 Translations
- 📊 New datasets
- 🤖 New models

---

## 📝 Citation

If you use Shifaa in your research, please cite:

```bibtex
@software{shifaa2025,
  title={Shifaa: Arabic Medical AI Platform},
  author={Ahmed Selim and Mariam Hassan and Ghada Saeed and Arwa Mohamed and Nour Ali and Hager Mohamed},
  year={2025},
  url={https://github.com/AhmedSeelim/shifaa},
  note={Datasets and models available at https://huggingface.co/Ahmed-Selem}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🌐 Links & Resources

- **GitHub:** [github.com/AhmedSeelim/shifaa](https://github.com/AhmedSeelim/shifaa)
- **HuggingFace:** [huggingface.co/Ahmed-Selem](https://huggingface.co/Ahmed-Selem)
- **PyPI:** [pypi.org/project/shifaa](https://pypi.org/project/shifaa)
- **Colab Notebook:** [Try Shifaa in Colab](https://colab.research.google.com/github/AhmedSeelim/shifaa/blob/main/Shifaa_Examples.ipynb)

---

## 💬 Support

- **Issues:** [GitHub Issues](https://github.com/AhmedSeelim/shifaa/issues)
- **Email:** ahmedselimmahmoud1@gmail.com

---


## 🙏 Acknowledgments

Special thanks to:
- The Arabic medical community for their invaluable feedback
- HuggingFace for hosting our datasets and models
- All contributors who made this project possible
- The MENA healthcare professionals using Shifaa

---

## 🌟 Star History

If you find Shifaa useful, please consider giving it a star ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=AhmedSeelim/shifaa&type=Date)](https://star-history.com/#AhmedSeelim/shifaa&Date)

---

<div align="center">

**Made with ❤️ for the MENA healthcare community**

</div>