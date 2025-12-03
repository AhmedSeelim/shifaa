# Shifaa Datasets Module

Easy access to official Shifaa datasets hosted on HuggingFace.

## 📊 Available Datasets

### 1. Shifaa Arabic Mental Health Consultations

A comprehensive dataset of real-world mental health consultations in Arabic.

**Statistics:**
- **Size:** 35,648 consultations
- **Main Specializations:** 7
- **Specific Diagnoses:** 123
- **Language:** Arabic
- **Quality:** Professional mental health responses

### 2. Shifaa Arabic Medical Consultations

High-quality medical consultations covering a wide range of specialties.

**Statistics:**
- **Size:** 84,422 consultations
- **Main Specializations:** 16
- **Hierarchical Diagnoses:** 585
- **Language:** Arabic
- **Average Answer Length:** 273 words
- **Quality:** No missing values, professionally reviewed

## 🚀 Quick Start

### Basic Usage

```python
from shifaa.datasets import load_shifaa_mental_dataset, load_shifaa_medical_dataset

# Load mental health dataset
mental_data = load_shifaa_mental_dataset()
print(f"Loaded {len(mental_data)} mental health consultations")

# Load medical consultations dataset
medical_data = load_shifaa_medical_dataset()
print(f"Loaded {len(medical_data)} medical consultations")

# Access individual consultations
first_consultation = mental_data[0]
print(first_consultation)
```

### Loading Specific Splits

```python
from shifaa.datasets import load_shifaa_medical_dataset

# Load only training split
train_data = load_shifaa_medical_dataset(split='train')

# Load only test split
test_data = load_shifaa_medical_dataset(split='test')
```

### Custom Cache Directory

```python
from shifaa.datasets import load_shifaa_mental_dataset

# Specify custom cache location
dataset = load_shifaa_mental_dataset(cache_dir='./my_cache')
```

## 📖 Dataset Information

### Get Dataset Info Without Loading

```python
from shifaa.datasets import get_dataset_info, list_available_datasets

# Get info for a specific dataset
mental_info = get_dataset_info("mental")
print(mental_info)
# Output: {'name': 'Shifaa Arabic Mental Health Consultations', 'size': 35648, ...}

medical_info = get_dataset_info("medical")
print(medical_info)

# List all available datasets
all_datasets = list_available_datasets()
for name, info in all_datasets.items():
    print(f"{name}: {info['size']} consultations")
```

## 🔍 Dataset Schema


```python
{
    'Consultation Number': int64,  # Unique identifier
    'Question Title': str,         # Brief title
    'Question': str,               # Patient's question/concern
    'Answer': str,                 # Professional response
    'Doctor Name': str,            # Responding professional
    'Hierarchical Diagnosis': str, # Hierarchical diagnosis path
    'Date of Answer': str,         # Consultation date
    'metadata': dict,              # Additional information
}
```

## 💡 Common Use Cases

### 1. Training Medical Chatbots

```python
from shifaa.datasets import load_shifaa_medical_dataset

dataset = load_shifaa_medical_dataset()

# Extract Q&A pairs for training
qa_pairs = []
for consultation in dataset:
    qa_pairs.append({
        'input': consultation['Question'],
        'output': consultation['Answer'],
        'Diagnosis': consultation['Hierarchical Diagnosis']
    })

# Use qa_pairs for fine-tuning your model
```

### 2. Building RAG Systems

```python
from shifaa.datasets import load_shifaa_medical_dataset

dataset = load_shifaa_medical_dataset()

# Create knowledge base for RAG
knowledge_base = []
for consultation in dataset:
    knowledge_base.append({
        'text': f"{consultation['Question Title']}\n{consultation['question']}",
        'answer': consultation['Answer'],
        'path': consultation['Hierarchical Diagnosis'],
        
    })

# Use knowledge_base with your RAG system
```

## 🌐 HuggingFace Links

- [Mental Health Dataset](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations)
- [Medical Consultations Dataset](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Medical_Consultations)


## 🤝 Contributing

Found an issue with the datasets? Please report it on our [GitHub Issues](https://github.com/yourusername/shifaa/issues).

## 📄 License

These datasets are released under appropriate licenses for medical data. Please check individual dataset cards on HuggingFace for specific licensing terms.