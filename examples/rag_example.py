"""
Shifaa Medical RAG Usage Examples

Demonstrates how to use the Medical RAG system.
"""

import os
from shifaa.rag import (
    MedicalRAGSystem,
    format_results_for_display,
    process_medical_data,
    get_insights_summary
)

import os
os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"  # Replace with your actual key

def example_1_basic_query():
    """Example 1: Basic query processing"""
    print("=" * 60)
    print("Example 1: Basic Query Processing")
    print("=" * 60)

    # Initialize RAG system
    print("\nInitializing Medical RAG System...")
    rag = MedicalRAGSystem()

    # Process a query
    query = "ما هي أعراض ارتجاع المريء؟"
    print(f"\nQuery: {query}")
    print("\nProcessing...")

    results = rag.process_query(query)

    if results:
        print("\n✓ Query processed successfully!")
        print(f"  - Found {len(results.specialties)} relevant specialties")
        print(f"  - Identified {len(results.topic_paths)} topic paths")
        print(f"  - Retrieved {len(results.consultations)} consultations")
        print(f"  - Extracted {len(results.insights)} insights")
    else:
        print("\n⚠ Query not recognized as medical")


def example_2_detailed_results():
    """Example 2: Exploring detailed results"""
    print("\n" + "=" * 60)
    print("Example 2: Detailed Results")
    print("=" * 60)

    rag = MedicalRAGSystem()
    query = "كيف أعالج الصداع المزمن؟"

    print(f"\nQuery: {query}")
    results = rag.process_query(query)

    if not results:
        print("⚠ No results found")
        return

    # Show specialties
    print("\n### Specialties ###")
    for specialty in results.specialties:
        print(f"\n{specialty.specialty}")
        print(f"  → {specialty.explanation}")

    # Show topic paths
    print("\n### Topic Paths ###")
    for topic in results.topic_paths:
        print(f"\n{topic.path}")
        print(f"  → {topic.explanation}")

    # Show consultations
    print("\n### Consultations ###")
    for i, consultation in enumerate(results.consultations, 1):
        print(f"\n[{i}] {consultation['metadata']['Question Title']}")
        print(f"    Similarity: {1 - consultation['distance']:.3f}")
        print(f"    Doctor: {consultation['metadata']['Doctor Name']}")

    # Show insights
    print("\n### Medical Insights ###")
    for i, insight in enumerate(results.insights, 1):
        print(f"\n[{i}] {insight.information}")
        print(f"    Relevance: {insight.relevance}")


def example_3_formatted_output():
    """Example 3: Using formatted output"""
    print("\n" + "=" * 60)
    print("Example 3: Formatted Output")
    print("=" * 60)

    rag = MedicalRAGSystem()
    query = "ما هي أسباب آلام المعدة؟"

    print(f"\nQuery: {query}")
    results = rag.process_query(query)

    if results:
        # Use built-in formatter
        formatted = format_results_for_display(results)
        print(formatted)


def example_4_processing_results():
    """Example 4: Processing and analyzing results"""
    print("\n" + "=" * 60)
    print("Example 4: Processing Results")
    print("=" * 60)

    rag = MedicalRAGSystem()
    query = "ما علاج الأرق؟"

    print(f"\nQuery: {query}")
    results = rag.process_query(query)

    if not results:
        return

    # Process results
    topics, consultations = process_medical_data(results)

    print("\n### Extracted Topics ###")
    for topic_name, topic_info in topics.items():
        print(f"\n{topic_name}")
        print(f"  {topic_info['explanation']}")

    print("\n### Consultations with Insights ###")
    for key, consultation in consultations.items():
        print(f"\n{consultation['title']}")
        print(f"  Similarity: {consultation['similarity']:.3f}")
        print(f"  Doctor: {consultation['doctor']}")
        print(f"  Insights: {len(consultation['information'])} found")

    # Get insights summary
    insights_summary = get_insights_summary(results)
    print("\n### Insights Summary ###")
    print(insights_summary)


def example_5_custom_configuration():
    """Example 5: Custom RAG configuration"""
    print("\n" + "=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)

    # Initialize with custom settings
    print("\nInitializing with custom settings...")
    rag = MedicalRAGSystem(
        llm_model_name="gemini-2.0-flash",
        temperature=0.0,
        n_results=5  # Retrieve more consultations
    )

    query = "ما هي أعراض السكري؟"
    print(f"\nQuery: {query}")

    results = rag.process_query(query)

    if results:
        print(f"\n✓ Retrieved {len(results.consultations)} consultations")
        print(f"✓ Extracted {len(results.insights)} insights")


def example_6_batch_processing():
    """Example 6: Processing multiple queries"""
    print("\n" + "=" * 60)
    print("Example 6: Batch Processing")
    print("=" * 60)

    rag = MedicalRAGSystem()

    queries = [
        "ما هي أعراض فقر الدم؟",
        "كيف أعالج حرقة المعدة؟",
        "ما سبب الدوخة المستمرة؟"
    ]

    print(f"\nProcessing {len(queries)} queries...")

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}] Query: {query}")
        results = rag.process_query(query)

        if results:
            print(f"    ✓ Specialties: {[s.specialty for s in results.specialties]}")
            print(f"    ✓ Insights: {len(results.insights)} found")
        else:
            print("    ⚠ No results")


def example_7_error_handling():
    """Example 7: Proper error handling"""
    print("\n" + "=" * 60)
    print("Example 7: Error Handling")
    print("=" * 60)

    try:
        # Check for API key
        if "GOOGLE_API_KEY" not in os.environ:
            print("\n⚠ GOOGLE_API_KEY not set!")
            print("Set it with: export GOOGLE_API_KEY='your-key-here'")
            return

        rag = MedicalRAGSystem()

        # Test with non-medical query
        query = "What's the weather today?"
        print(f"\nTesting non-medical query: {query}")

        results = rag.process_query(query)

        if results:
            print("✓ Query processed (identified as medical)")
        else:
            print("✓ Query correctly identified as non-medical")

    except Exception as e:
        print(f"\n⚠ Error occurred: {str(e)}")
        print("Make sure all dependencies are installed and configured")


def main():
    """Run all examples"""
    print("\n🏥 Shifaa Medical RAG Examples")
    print("=" * 60)

    # Check for API key
    if "GOOGLE_API_KEY" not in os.environ:
        print("\n⚠ WARNING: GOOGLE_API_KEY not set!")
        print("The RAG system requires a Google API key to function.")
        print("Set it with: export GOOGLE_API_KEY='your-key-here'")
        print("\nSome examples may fail without it.\n")

    examples = [
        example_1_basic_query,
        example_2_detailed_results,
        example_3_formatted_output,
        example_4_processing_results,
        example_5_custom_configuration,
        example_6_batch_processing,
        example_7_error_handling,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n⚠ Error in {example.__name__}: {str(e)}")

    print("\n" + "=" * 60)
    print("✓ Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()