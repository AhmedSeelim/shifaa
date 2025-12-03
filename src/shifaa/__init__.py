"""
Shifaa - Arabic Medical AI Platform

Shifaa revolutionizes Arabic medical AI by providing:
- High-quality Arabic medical datasets
- Advanced RAG systems for medical information retrieval
- Vision-based medical analysis tools (coming soon)

Modules:
    shifaa.datasets  - Access to curated Arabic medical datasets
    shifaa.rag       - Medical Retrieval-Augmented Generation system
    shifaa.vision    - Medical vision analysis (coming soon)

Quick Start:
    >>> # Load datasets
    >>> from shifaa.datasets import load_shifaa_medical_dataset
    >>> dataset = load_shifaa_medical_dataset()

    >>> # Use Medical RAG
    >>> from shifaa.rag import MedicalRAGSystem
    >>> rag = MedicalRAGSystem()
    >>> results = rag.process_query("ما هي أعراض السكري؟")

Documentation: https://shifaa.readthedocs.io
GitHub: https://github.com/yourusername/shifaa
HuggingFace: https://huggingface.co/Ahmed-Selem
"""

__version__ = '0.1.0'
__author__ = 'Ahmed Selem'
__email__ = 'ahmed.selem@example.com'

# Import main components for convenience
from . import datasets
from . import rag
from . import vision

__all__ = [
    'datasets',
    'rag',
    'vision',
]