"""
Utility Functions for RAG System

Helper functions for processing and formatting RAG results.
"""

from typing import Dict, List, Any, Tuple


def extract_topic_from_path(path: str) -> str:
    """
    Extract topic from path string.

    If last element is 'أخرى' (Other), return second-to-last element.
    Otherwise, return last element.

    Args:
        path: Path string with elements separated by '>'

    Returns:
        Extracted topic name

    Example:
        >>> extract_topic_from_path("Cardiology > Heart Disease > Other")
        'Heart Disease'
        >>> extract_topic_from_path("Cardiology > Heart Disease > Arrhythmia")
        'Arrhythmia'
    """
    elements = [elem.strip() for elem in path.split('>')]

    if len(elements) == 0:
        return "Unknown"

    if elements[-1] == 'أخرى' and len(elements) > 1:
        return elements[-2]

    return elements[-1]


def generate_consultation_key(consultation: Dict[str, Any]) -> str:
    """
    Generate a unique key for consultation to handle duplicates.

    Uses text content as primary identifier since titles can be empty.

    Args:
        consultation: Consultation dictionary with text and metadata

    Returns:
        Unique key string
    """
    title = consultation.get('metadata', {}).get('Question Title', '').strip()
    text = consultation.get('text', '').strip()

    # If title is empty or very short, use first 100 chars of text
    if not title or len(title) < 3:
        text_key = text[:100].replace('\n', ' ').strip()
        return text_key if text_key else f"consultation_{id(consultation)}"

    return title


def process_medical_data(results) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process medical RAG results to extract topics and match insights with consultations.

    Args:
        results: Medical RAG system results object (MedicalRAGOutput)

    Returns:
        Tuple of (topics_dict, consultations_dict)
        - topics_dict: Dictionary of unique topics with explanations
        - consultations_dict: Dictionary of consultations that have insights

    Example:
        >>> topics, consultations = process_medical_data(results)
        >>> for topic_name, topic_info in topics.items():
        ...     print(f"{topic_name}: {topic_info['explanation']}")
    """
    # Extract topics dictionary
    topics = {}
    for topic_path in results.topic_paths:
        topic_name = extract_topic_from_path(topic_path.path)
        if topic_name and topic_name not in topics:
            topics[topic_name] = {
                'topic': topic_name,
                'explanation': topic_path.explanation
            }

    # Create consultations dictionary - only for consultations with insights
    consultations = {}

    # Create mapping of all consultations first
    all_consultations = {}
    for consultation in results.consultations:
        consultation_key = generate_consultation_key(consultation)

        # Skip duplicates
        if consultation_key in all_consultations:
            continue

        title = consultation.get('metadata', {}).get('Question Title', '').strip()

        all_consultations[consultation_key] = {
            'key': consultation_key,
            'title': title if title else 'Untitled Consultation',
            'text': consultation.get('text', ''),
            'similarity': 1 - consultation.get('distance', 0),
            'doctor': consultation.get('metadata', {}).get('Doctor Name', 'Unknown'),
            'path': consultation.get('metadata', {}).get('Path', ''),
            'topic': extract_topic_from_path(
                consultation.get('metadata', {}).get('Path', '')
            ),
            'answer': consultation.get('metadata', {}).get('Answer', '')
        }

    # Only add consultations that have insights
    for insight in results.insights:
        insight_title = getattr(insight, 'consultation_title', '').strip()

        # Find matching consultation
        matching_key = None

        if insight_title and insight_title in all_consultations:
            matching_key = insight_title
        else:
            # Try fuzzy matching
            for key, consultation in all_consultations.items():
                if (insight_title and
                        (insight_title == consultation['title'] or
                         insight_title in consultation['text'][:200] or
                         consultation['title'] in insight_title)):
                    matching_key = key
                    break

        if matching_key:
            if matching_key not in consultations:
                consultations[matching_key] = all_consultations[matching_key].copy()
                consultations[matching_key]['information'] = []
                consultations[matching_key]['relevance'] = []

            # Add insight information
            consultations[matching_key]['information'].append(
                getattr(insight, 'information', '')
            )
            consultations[matching_key]['relevance'].append(
                getattr(insight, 'relevance', '')
            )

    return topics, consultations


def get_insights_summary(results) -> str:
    """
    Extract all insights information in a compact format.

    Args:
        results: Medical RAG system results object

    Returns:
        Formatted string containing medical insights

    Example:
        >>> summary = get_insights_summary(results)
        >>> print(summary)
    """
    medical_info = {}

    for i, insight in enumerate(results.insights, 1):
        key = f"insight_{i}"
        medical_info[key] = getattr(insight, 'information', '')

    return str(medical_info)


def get_complete_summary(results) -> Tuple[Dict, Dict, str]:
    """
    Get complete summary including topics, consultations, and insights.

    Args:
        results: Medical RAG system results object

    Returns:
        Tuple of (topics, consultations, insights_summary_string)
    """
    topics, consultations = process_medical_data(results)
    insights_summary = get_insights_summary(results)

    return topics, consultations, insights_summary


def format_results_for_display(results) -> str:
    """
    Format RAG results into a readable string for display.

    Args:
        results: Medical RAG system results object

    Returns:
        Formatted string with all results
    """
    output = []

    output.append("=" * 60)
    output.append("MEDICAL RAG RESULTS")
    output.append("=" * 60)

    # Specialties
    output.append("\n### SPECIALTIES ###")
    for specialty in results.specialties:
        output.append(f"\n• {specialty.specialty}")
        output.append(f"  {specialty.explanation}")

    # Topic Paths
    output.append("\n\n### TOPIC PATHS ###")
    for topic in results.topic_paths:
        output.append(f"\n• {topic.path}")
        output.append(f"  {topic.explanation}")

    # Consultations
    output.append("\n\n### CONSULTATIONS ###")
    for i, consultation in enumerate(results.consultations, 1):
        output.append(f"\n[{i}] {consultation['metadata']['Question Title']}")
        output.append(f"    Similarity: {1 - consultation['distance']:.3f}")
        output.append(f"    Doctor: {consultation['metadata']['Doctor Name']}")
        output.append(f"    Path: {consultation['metadata']['Path']}")

    # Insights
    output.append("\n\n### MEDICAL INSIGHTS ###")
    for i, insight in enumerate(results.insights, 1):
        output.append(f"\n[{i}] {insight.information}")
        output.append(f"    Relevance: {insight.relevance}")
        output.append(f"    Source: {insight.consultation_title}")

    output.append("\n" + "=" * 60)

    return "\n".join(output)