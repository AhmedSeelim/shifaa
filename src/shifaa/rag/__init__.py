"""
Shifaa RAG Module

Medical Retrieval-Augmented Generation system for Arabic medical queries.

Features:
- Automatic specialty detection
- Hierarchical topic path identification
- Semantic search over 84K+ medical consultations
- Context-aware medical insights extraction

Example:
    >>> from shifaa.rag import MedicalRAGSystem
    >>> rag = MedicalRAGSystem()
    >>> results = rag.process_query("ما هي أعراض السكري؟")
    >>> print(results.specialties)
    >>> print(results.insights)
"""

from .medical_rag import (
    MedicalRAGSystem,
    MedicalRAGOutput,
    SpecialtyReason,
    TopicReason,
    MedicalFact,
)
from .document_processor import MedicalDocumentProcessor
from .vector_db import VectorDBManager, download_vector_db
from .utils import (
    process_medical_data,
    get_insights_summary,
    get_complete_summary,
    format_results_for_display,
)

__all__ = [
    # Main RAG System
    'MedicalRAGSystem',

    # Output Models
    'MedicalRAGOutput',
    'SpecialtyReason',
    'TopicReason',
    'MedicalFact',

    # Document Processing
    'MedicalDocumentProcessor',

    # Vector Database
    'VectorDBManager',
    'download_vector_db',

    # Utilities
    'process_medical_data',
    'get_insights_summary',
    'get_complete_summary',
    'format_results_for_display',
]

__version__ = '0.1.0'