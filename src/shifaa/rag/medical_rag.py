"""
Medical RAG System

A sequential chain system for medical query processing using LangChain.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import os
import json
from pathlib import Path
import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

from .document_processor import MedicalDocumentProcessor
from .vector_db import VectorDBManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for structured outputs
class SpecialtyReason(BaseModel):
    specialty: str
    explanation: str


class SpecialtyOutput(BaseModel):
    relevant_specialties: List[SpecialtyReason]


class TopicReason(BaseModel):
    path: str
    explanation: str


class RelevantTopics(BaseModel):
    relevant_topics: List[TopicReason]


class MedicalFact(BaseModel):
    information: str = Field(..., description="Extracted medical advice or fact.")
    relevance: str = Field(..., description="Why this information is relevant to the user's query.")
    consultation_title: str = Field(
        ...,
        description="Question title from the consultation."
    )


class ExtractedMedicalInsights(BaseModel):
    extracted_info: List[MedicalFact] = Field(
        ...,
        description="A list of relevant insights extracted from consultations."
    )


class MedicalRAGOutput(BaseModel):
    """Complete output model for the entire RAG pipeline."""
    specialties: List[SpecialtyReason]
    topic_paths: List[TopicReason]
    consultations: List[Dict[str, Any]]
    insights: List[MedicalFact]


class MedicalRAGSystem:
    """
    Sequential chain system for medical query processing.

    Pipeline:
    1. Detects relevant medical specialties
    2. Identifies relevant medical topic paths
    3. Retrieves related consultations
    4. Extracts important medical insights
    """

    def __init__(
            self,
            llm_model_name: str = "gemini-2.0-flash",
            temperature: float = 0.0,
            db_path: Optional[str] = None,
            categories_json_path: Optional[str] = None,
            n_results: int = 3,
            google_api_key: Optional[str] = None,
            auto_download_db: bool = True
    ):
        """
        Initialize the Medical RAG System.

        Args:
            llm_model_name: The LLM model to use
            temperature: Temperature setting for the LLM
            db_path: Path to the vector database (auto-downloads if not provided)
            categories_json_path: Path to medical categories JSON
            n_results: Number of consultation results to retrieve
            google_api_key: Google API key (can also be set via GOOGLE_API_KEY env var)
            auto_download_db: Automatically download vector DB if not present
        """
        # Set up Google API key
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        elif "GOOGLE_API_KEY" not in os.environ:
            logger.warning(
                "GOOGLE_API_KEY not found. "
                "Please set it via environment variable or pass it to the constructor."
            )

        # Handle vector database
        if db_path is None:
            logger.info("No db_path provided. Managing vector database automatically...")
            db_manager = VectorDBManager()
            db_path = str(db_manager.get_db_path(auto_download=auto_download_db))

        # Load medical categories
        if categories_json_path is None:
            # Try to find it in the package directory
            package_dir = Path(__file__).parent
            categories_json_path = package_dir / "medical_categories.json"

        if not Path(categories_json_path).exists():
            raise FileNotFoundError(
                f"Medical categories file not found at {categories_json_path}. "
                "Please provide a valid path."
            )

        try:
            with open(categories_json_path, "r", encoding="utf-8") as f:
                self.path_dec = json.load(f)
            logger.info(f"Loaded medical categories from: {categories_json_path}")
        except json.JSONDecodeError:
            raise ValueError(
                f"Invalid JSON in medical categories file at {categories_json_path}"
            )

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model_name,
            temperature=temperature
        )
        self.n_results = n_results

        # Initialize document processor
        try:
            self.document_processor = MedicalDocumentProcessor(db_path=db_path)
            logger.info("✓ Medical RAG System initialized successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MedicalDocumentProcessor: {str(e)}"
            )

        # Initialize all chains
        self._init_specialty_chain()
        self._init_topic_path_chain()
        self._init_insights_chain()

    def _init_specialty_chain(self):
        """Initialize the specialty detection chain."""
        parser = PydanticOutputParser(pydantic_object=SpecialtyOutput)

        prompt = PromptTemplate(
            template="""
You are a medical assistant AI that helps determine which medical specialty a user's query belongs to.

Below is a list of main medical specialties. Each includes a brief explanation of the types of symptoms, diseases, or medical issues it covers.

Your task: Given a medical question, select the most relevant specialty or specialties.

Main Specialties:

1. Head Diseases & ENT Diseases: Dizziness, headaches, general head pain, ENT issues
2. Eye Diseases: Vision problems, blindness, conjunctivitis, retinal problems
3. Oral & Dental Diseases: Gum infections, tooth decay, braces, oral ulcers
4. Cardio-Respiratory Diseases: Heart, blood vessels, lungs, breathing issues, asthma
5. Abdominal Diseases: Liver, spleen, pancreas, esophagus, stomach, intestines, colon
6. Bone Diseases: Bone pain, osteoporosis, cartilage issues, spine, joints
7. Muscle Diseases: Muscle weakness, inflammation, tears, cramps
8. Nervous System Diseases: Stroke, epilepsy, memory problems, nerve pain
9. Hair & Scalp Disorders: Baldness, dandruff, early graying, scalp infections
10. Skin Diseases: Allergies, rashes, itching, acne, boils, scars
11. Blood & Tumors: Cancers, tumors, hypertension, diabetes, cholesterol, anemia
12. Glands & Hormones: Thyroid, hormonal imbalances, adrenal or pituitary issues
13. Urinary System Diseases: Kidney diseases, UTI, bladder problems
14. Genital System Diseases: Sexual health issues, fertility problems
15. Gynecology: Menstrual disorders, ovarian cysts, uterus problems
16. Obstetrics and Pregnancy Problems: Fertility, pregnancy hormones, fetal problems
17. Pediatrics: Newborn and child care, feeding, vaccination
18. General Surgery & Cosmetic Surgery: Burns, surgeries, liposuction
19. Physical Health: Obesity, underweight, fitness, diet plans
20. Alternative Medicine: Herbal treatments, cupping, acupuncture
21. Miscellaneous Medical Topics: General health advice, immunity
22. Medications & Products: Medicine use, side effects
23. None

Important:
- If the query is not medical, return "None"
- Write specialty names exactly as shown in English
- Explanation must be in the same language as the query
- Choose only 1-2 most relevant specialties

User Query:
{query}

{format_instructions}
""",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        self.specialty_chain = prompt | self.llm | parser

    def _init_topic_path_chain(self):
        """Initialize the topic path detection chain."""
        parser = PydanticOutputParser(pydantic_object=RelevantTopics)

        prompt = PromptTemplate(
            template="""
You are a medical assistant AI that identifies relevant medical topics for a query.

You will receive:
- A user query
- A list of detailed medical topic paths under relevant specialties

Your task:
- Analyze the query
- Select only topics directly related to the query
- Explain briefly why each is relevant
- Response must be in the same language as the query

Query:
{query}

Topic Paths:
{topic_paths}

{format_instructions}
""",
            input_variables=["query", "topic_paths"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        self.topic_path_chain = prompt | self.llm | parser

    def _init_insights_chain(self):
        """Initialize the medical insights extraction chain."""
        parser = PydanticOutputParser(pydantic_object=ExtractedMedicalInsights)

        prompt = PromptTemplate(
            template="""
You are a medical assistant AI analyzing past medical consultations.

Given:
- A user query
- Medical consultations with titles and answers

Your task:
- Extract relevant medical facts or advice
- For each insight include:
  - The medical information
  - Why it's relevant
  - The consultation title it came from
- Response must be in the same language as the query

{format_instructions}

User Query:
{query}

Consultations:
{consultations}
""",
            input_variables=["query", "consultations"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        self.insights_chain = prompt | self.llm | parser

    def _collect_related_topic_paths(
            self,
            related_main_specialties: List[str]
    ) -> List[str]:
        """Collect topic paths related to given specialties."""
        related_topic_paths = []
        for specialty in related_main_specialties:
            if specialty in self.path_dec:
                related_topic_paths.extend(self.path_dec[specialty])
        return related_topic_paths

    def _format_consultations(
            self,
            retriever_results: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Format retriever results for the insights chain."""
        formatted = []
        for result in retriever_results:
            formatted.append({
                "consultation": result["text"],
                "answer": result["metadata"].get("Answer", "")
            })
        return formatted

    def process_query(self, query: str) -> Optional[MedicalRAGOutput]:
        """
        Process a medical query through the entire RAG pipeline.

        Args:
            query: The medical query text

        Returns:
            MedicalRAGOutput object with all results, or None if not medical

        Example:
            >>> rag = MedicalRAGSystem()
            >>> results = rag.process_query("ما هي أعراض السكري؟")
            >>> print(results.specialties)
            >>> print(results.insights)
        """
        try:
            # Step 1: Identify relevant specialties
            specialty_output = self.specialty_chain.invoke({"query": query})
            specialties = [s.specialty for s in specialty_output.relevant_specialties]

            # Check if query is medical
            if "None" in specialties or len(specialties) == 0:
                logger.info("Query identified as non-medical")
                return None

            # Step 2: Collect related topic paths
            related_topic_paths = self._collect_related_topic_paths(specialties)

            # Step 3: Identify relevant topic paths
            topic_paths_output = self.topic_path_chain.invoke({
                "query": query,
                "topic_paths": "\n".join(related_topic_paths)
            })
            relevant_paths = [topic.path for topic in topic_paths_output.relevant_topics]

            # Step 4: Retrieve relevant consultations
            consultations_results = []
            try:
                consultations_results = self.document_processor.retrieve_documents(
                    query=query,
                    n_results=self.n_results,
                    path_filter=relevant_paths
                )
            except Exception as e:
                logger.warning(f"Error in document retrieval: {str(e)}")

            # Step 5: Extract medical insights
            formatted_consultations = self._format_consultations(consultations_results)

            insights_output = ExtractedMedicalInsights(extracted_info=[])
            if formatted_consultations:
                insights_output = self.insights_chain.invoke({
                    "query": query,
                    "consultations": formatted_consultations
                })

            # Return comprehensive output
            return MedicalRAGOutput(
                specialties=specialty_output.relevant_specialties,
                topic_paths=topic_paths_output.relevant_topics,
                consultations=consultations_results,
                insights=insights_output.extracted_info
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise RuntimeError(f"Error processing query: {str(e)}")

    def __str__(self):
        """String representation of the Medical RAG System."""
        return "Medical RAG System (Specialties → Topic Paths → Consultations → Insights)"