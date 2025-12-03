"""
Medical Document Processor

Handles document processing and retrieval from ChromaDB vector database.
"""

from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer
import chromadb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingFunction:
    """Custom embedding function class for ChromaDB that uses SentenceTransformer."""

    def __init__(self, model_name: str = "Alibaba-NLP/gte-multilingual-base"):
        """
        Initialize the embedding function.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True
        ).to(self.device)

        logger.info(f"Loaded embedding model: {model_name}")

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            input: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(input, convert_to_tensor=False)
        return embeddings.tolist()


class MedicalDocumentProcessor:
    """Process and retrieve medical Q&A documents from ChromaDB."""

    def __init__(
            self,
            model_name: str = "Alibaba-NLP/gte-multilingual-base",
            db_path: str = None
    ):
        """
        Initialize the document processor.

        Args:
            model_name: Name of the embedding model
            db_path: Path to the ChromaDB database
        """
        if db_path is None:
            raise ValueError("db_path must be provided")

        # Initialize the embedding function
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name)

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(db_path)
        logger.info(f"Connected to ChromaDB at: {db_path}")

    def retrieve_documents(
            self,
            query: str,
            collection_name: str = "medical_qa_arabic",
            path_filter: Optional[List[str]] = None,
            n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents by query with optional path filtering.

        Args:
            query: The search query
            collection_name: Name of the ChromaDB collection
            path_filter: Optional list of paths to filter by
            n_results: Number of results to return

        Returns:
            List of matching documents with metadata and distances

        Example:
            >>> processor = MedicalDocumentProcessor(db_path="./vector_db")
            >>> results = processor.retrieve_documents(
            ...     query="ما هي أعراض السكري؟",
            ...     n_results=3
            ... )
            >>> for result in results:
            ...     print(result['text'])
        """
        try:
            # Get the collection
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )

            # Prepare where filter if path is specified
            where_filter = None
            if path_filter:
                if isinstance(path_filter, list) and len(path_filter) > 0:
                    where_filter = {"Path": {"$in": path_filter}}
                elif isinstance(path_filter, str):
                    where_filter = {"Path": {"$eq": path_filter}}

            # Query the collection
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def get_all_paths(self, collection_name: str = "medical_qa_arabic") -> List[str]:
        """
        Get all unique paths in the collection.

        Args:
            collection_name: Name of the ChromaDB collection

        Returns:
            Sorted list of unique paths
        """
        try:
            collection = self.chroma_client.get_collection(name=collection_name)

            # Get all documents
            all_docs = collection.get()

            # Extract unique paths
            paths = set()
            if all_docs["metadatas"]:
                for metadata in all_docs["metadatas"]:
                    if "Path" in metadata:
                        paths.add(metadata["Path"])

            return sorted(list(paths))

        except Exception as e:
            logger.error(f"Error getting paths: {str(e)}")
            raise

    def get_collection_stats(
            self,
            collection_name: str = "medical_qa_arabic"
    ) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Args:
            collection_name: Name of the ChromaDB collection

        Returns:
            Dictionary containing collection statistics
        """
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            count = collection.count()

            return {
                "collection_name": collection_name,
                "total_documents": count,
                "unique_paths": len(self.get_all_paths(collection_name))
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise