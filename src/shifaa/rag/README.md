# Shifaa Medical RAG System

Advanced Retrieval-Augmented Generation (RAG) system for Arabic medical consultations.

## 🎯 Overview

The Shifaa Medical RAG system provides intelligent medical information retrieval through a four-stage pipeline:

```
Query → Specialty Detection → Topic Paths → Consultation Retrieval → Insight Extraction
```

### Key Features

- **Automatic Specialty Detection**: Identifies relevant medical specialties from 23 categories
- **Hierarchical Topic Navigation**: Pinpoints specific medical topics from 585 diagnoses
- **Semantic Search**: Retrieves similar consultations from 84K+ medical cases
- **Insight Extraction**: Distills actionable medical information from retrieved consultations
- **Multi-lingual Support**: Primary support for Arabic with multilingual capabilities
- **Auto-Download**: Automatically manages vector database downloads

## 🚀 Quick Start

### Basic Usage

```python
from shifaa.rag import MedicalRAGSystem

# Initialize the system (auto-downloads vector DB if needed)
rag = MedicalRAGSystem()

# Process a medical query
query = "ما هي أعراض ارتجاع المريء؟"
results = rag.process_query(query)

# Access results
print("Specialties:", [s.specialty for s in results.specialties])
print("Topics:", [t.path for t in results.topic_paths])
print("Insights:", [i.information for i in results.insights])
```

### With Google API Key

```python
from shifaa.rag import MedicalRAGSystem

rag = MedicalRAGSystem(
    google_api_key="your-api-key-here"
)

results = rag.process_query("ما علاج الصداع المزمن؟")
```

## 📦 Installation & Setup

### Prerequisites

```bash
pip install shifaa
```

### Google API Key Setup

The RAG system requires a Google API key for the LLM:

**Option 1: Environment Variable (Recommended)**
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

**Option 2: Pass to Constructor**
```python
rag = MedicalRAGSystem(google_api_key="your-api-key-here")
```

**Option 3: Config File**
Create `config.py`:
```python
GOOGLE_API_KEY = "your-api-key-here"
```

### Vector Database Management

The vector database is automatically downloaded on first use (~500MB).

**Default Location:**
```
~/.shifaa/vector_db/medical_qa_chroma_db/
```

**Custom Location:**
```python
rag = MedicalRAGSystem(db_path="./my_vector_db")
```

**Manual Download:**
```bash
# Using CLI
shifaa-download-db --path ./custom_path

# Using Python
from shifaa.rag import download_vector_db
db_path = download_vector_db(extract_path="./custom_path")
```

**Clear Cache:**
```python
from shifaa.rag import VectorDBManager

manager = VectorDBManager()
manager.clear_cache()  # Removes downloaded database
```

## 🏗️ System Architecture

### Pipeline Stages

#### 1. Specialty Detection
Identifies relevant medical specialties from the query.

```python
results = rag.process_query("عندي ألم في الصدر")
for specialty in results.specialties:
    print(f"{specialty.specialty}: {specialty.explanation}")
# Output: Cardio-Respiratory Diseases: يتعلق بألم الصدر...
```

#### 2. Topic Path Identification
Narrows down to specific medical topics within specialties.

```python
for topic in results.topic_paths:
    print(f"{topic.path}")
# Output: Cardio-Respiratory Diseases > Heart Diseases > Chest Pain
```

#### 3. Consultation Retrieval
Retrieves similar consultations using semantic search.

```python
for consultation in results.consultations:
    print(f"Title: {consultation['metadata']['Question Title']}")
    print(f"Similarity: {1 - consultation['distance']:.3f}")
    print(f"Doctor: {consultation['metadata']['Doctor Name']}")
```

#### 4. Insight Extraction
Extracts relevant medical facts from retrieved consultations.

```python
for insight in results.insights:
    print(f"Info: {insight.information}")
    print(f"Relevance: {insight.relevance}")
    print(f"Source: {insight.consultation_title}")
```

## 🔧 Advanced Configuration

### Custom LLM Settings

```python
rag = MedicalRAGSystem(
    llm_model_name="gemini-2.0-flash-exp",  # Model name
    temperature=0.0,                         # Creativity (0-1)
    n_results=5                              # Number of consultations to retrieve
)
```

### Custom Categories File

```python
rag = MedicalRAGSystem(
    categories_json_path="./my_categories.json"
)
```

### Disable Auto-Download

```python
rag = MedicalRAGSystem(
    db_path="./existing_db",
    auto_download_db=False  # Raises error if DB not found
)
```

## 📊 Working with Results

### Result Object Structure

```python
results = rag.process_query(query)

# MedicalRAGOutput object contains:
results.specialties     # List[SpecialtyReason]
results.topic_paths     # List[TopicReason]
results.consultations   # List[Dict]
results.insights        # List[MedicalFact]
```

### Processing Results

```python
from shifaa.rag import process_medical_data, get_insights_summary

# Extract topics and consultations
topics, consultations = process_medical_data(results)

# Get insights summary
insights_text = get_insights_summary(results)

# Format for display
from shifaa.rag import format_results_for_display
display_text = format_results_for_display(results)
print(display_text)
```

### Filtering Consultations

```python
# Filter by specialty
Bone_consultations = [
    c for c in results.consultations 
    if "Bone Diseases" in c['metadata']['Path']
]

# Filter by similarity threshold
high_similarity = [
    c for c in results.consultations 
    if (1 - c['distance']) > 0.8
]
```

## 💡 Use Cases

### 1. Medical Chatbot Backend

```python
from shifaa.rag import MedicalRAGSystem

class MedicalChatbot:
    def __init__(self):
        self.rag = MedicalRAGSystem()
    
    def answer_query(self, user_query: str) -> str:
        results = self.rag.process_query(user_query)
        
        if not results:
            return "عذراً، لا أستطيع فهم السؤال. هل هو سؤال طبي؟"
        
        # Compile answer from insights
        answer_parts = []
        for insight in results.insights:
            answer_parts.append(insight.information)
        
        return "\n\n".join(answer_parts)

chatbot = MedicalChatbot()
response = chatbot.answer_query("ما هي أعراض السكري؟")
print(response)
```

### 2. Medical Knowledge Search

```python
from shifaa.rag import MedicalRAGSystem

def search_medical_knowledge(query: str, n_results: int = 5):
    rag = MedicalRAGSystem(n_results=n_results)
    results = rag.process_query(query)
    
    if not results:
        return []
    
    # Return relevant consultations
    return [{
        'title': c['metadata']['Question Title'],
        'answer': c['metadata']['Answer'],
        'doctor': c['metadata']['Doctor Name'],
        'similarity': 1 - c['distance']
    } for c in results.consultations]

results = search_medical_knowledge("علاج الصداع النصفي", n_results=3)
for result in results:
    print(f"\n{result['title']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Answer: {result['answer'][:200]}...")
```

### 3. Symptom Analysis

```python
from shifaa.rag import MedicalRAGSystem

def analyze_symptoms(symptoms: str) -> dict:
    rag = MedicalRAGSystem()
    results = rag.process_query(symptoms)
    
    if not results:
        return {"error": "Could not analyze symptoms"}
    
    return {
        "possible_specialties": [s.specialty for s in results.specialties],
        "related_topics": [t.path for t in results.topic_paths],
        "relevant_cases": len(results.consultations),
        "key_insights": [i.information for i in results.insights]
    }

analysis = analyze_symptoms("دوخة وصداع مستمر")
print(f"Specialties: {analysis['possible_specialties']}")
print(f"Topics: {analysis['related_topics']}")
```

### 4. Batch Processing

```python
from shifaa.rag import MedicalRAGSystem
import json

def process_queries_batch(queries: list) -> list:
    rag = MedicalRAGSystem()
    results_list = []
    
    for query in queries:
        result = rag.process_query(query)
        if result:
            results_list.append({
                "query": query,
                "specialties": [s.specialty for s in result.specialties],
                "insights_count": len(result.insights)
            })
    
    return results_list

queries = [
    "ما هي أسباب آلام المعدة؟",
    "كيف أعالج الأرق؟",
    "ما هي أعراض فقر الدم؟"
]

batch_results = process_queries_batch(queries)
print(json.dumps(batch_results, ensure_ascii=False, indent=2))
```

## 🔍 Troubleshooting

### Common Issues

**Issue: "GOOGLE_API_KEY not found"**
```python
# Solution: Set the API key
import os
os.environ["GOOGLE_API_KEY"] = "your-key-here"
```

**Issue: Vector database download fails**
```python
# Solution: Download manually
from shifaa.rag import download_vector_db
db_path = download_vector_db(force=True)
```

**Issue: Out of memory with large batches**
```python
# Solution: Reduce n_results
rag = MedicalRAGSystem(n_results=2)
```

**Issue: Slow first query**
```python
# Note: First query loads the model (normal)
# Subsequent queries will be faster
```

## 📈 Performance Tips

1. **Reuse RAG Instance**: Initialize once, query multiple times
2. **Adjust n_results**: Balance between relevance and speed
3. **Use GPU**: Significant speedup for embeddings (automatic if available)
4. **Cache Frequently**: Vector DB is automatically cached
5. **Batch Queries**: Process multiple queries in one session

## 📝 Citation

```bibtex
@software{shifaa2025,
  title={Shifaa: Arabic Medical AI Platform},
  author={Ahmed Selim and Mariam Hassan and Ghada Saeed and Arwa Mohamed and Nour Ali and Hager Mohamed},
  year={2025},
  url={https://github.com/AhmedSeelim/shifaa},
  note={Datasets and models available at https://huggingface.co/Ahmed-Selem}
}
```

## 🤝 Contributing

Found a bug or have a feature request? Please open an issue on [GitHub](https://github.com/yourusername/shifaa/issues).

## ⚠️ Limitations

- **Mental Health RAG**: Not yet implemented (coming soon)
- **Language**: Optimized for Arabic, other languages may have reduced performance
- **Medical Advice**: This is an information retrieval system, not a replacement for professional medical consultation
- **API Key Required**: Requires Google API key for LLM functionality

## 📄 License

This module is part of the Shifaa package and is licensed under the MIT License.