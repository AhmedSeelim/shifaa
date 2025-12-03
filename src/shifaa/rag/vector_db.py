"""
Vector Database Management Module

Handles downloading and managing the Shifaa Medical RAG Vector Database.
"""

import os
import zipfile
import logging
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBConfig:
    """Configuration for Vector Database management."""

    REPO_ID = "Ahmed-Selem/Shifaa_Medical_RAG_VectorDB"
    FILENAME = "medical_qa_chroma_db.zip"
    REPO_TYPE = "dataset"
    DEFAULT_EXTRACT_PATH = Path.home() / ".shifaa" / "vector_db"
    EXTRACTED_FOLDER_NAME = "medical_qa_chroma_db"


class VectorDBManager:
    """Manages vector database download, extraction, and caching."""

    def __init__(self, extract_path: Optional[Path] = None):
        """
        Initialize the Vector DB Manager.

        Args:
            extract_path: Custom path to extract the database.
                         Defaults to ~/.shifaa/vector_db
        """
        self.extract_path = extract_path or VectorDBConfig.DEFAULT_EXTRACT_PATH
        self.extract_path = Path(self.extract_path)
        self.db_path = self.extract_path / VectorDBConfig.EXTRACTED_FOLDER_NAME

    def is_downloaded(self) -> bool:
        """
        Check if the vector database is already downloaded and extracted.

        Returns:
            True if database exists and appears valid, False otherwise
        """
        if not self.db_path.exists():
            return False

        # Check if it contains expected ChromaDB files
        expected_files = ["chroma.sqlite3"]
        for file in expected_files:
            if not (self.db_path / file).exists():
                logger.warning(f"Database appears incomplete. Missing: {file}")
                return False

        return True

    def download_and_extract(self, force: bool = False) -> Path:
        """
        Download and extract the vector database from HuggingFace.

        Args:
            force: If True, redownload even if already exists

        Returns:
            Path to the extracted database directory

        Raises:
            RuntimeError: If download or extraction fails
        """
        # Check if already downloaded
        if self.is_downloaded() and not force:
            logger.info(f"✓ Vector database already exists at: {self.db_path}")
            return self.db_path

        try:
            logger.info("Downloading Shifaa Medical Vector Database from HuggingFace...")
            logger.info(f"Repository: {VectorDBConfig.REPO_ID}")

            # Create extract directory if it doesn't exist
            self.extract_path.mkdir(parents=True, exist_ok=True)

            # Download the zip file
            zip_path = hf_hub_download(
                repo_id=VectorDBConfig.REPO_ID,
                filename=VectorDBConfig.FILENAME,
                repo_type=VectorDBConfig.REPO_TYPE,
                force_download=force
            )

            logger.info(f"✓ Download complete: {zip_path}")
            logger.info(f"Extracting to: {self.extract_path}")

            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files for progress
                file_list = zip_ref.namelist()
                logger.info(f"Extracting {len(file_list)} files...")

                # Extract all files
                zip_ref.extractall(self.extract_path)

            logger.info(f"✓ Extraction complete")
            logger.info(f"✓ Vector database ready at: {self.db_path}")

            # Verify extraction
            if not self.is_downloaded():
                raise RuntimeError("Database extraction appears to be incomplete")

            return self.db_path

        except Exception as e:
            logger.error(f"Failed to download/extract vector database: {str(e)}")
            raise RuntimeError(f"Vector database setup failed: {str(e)}")

    def get_db_path(self, auto_download: bool = True) -> Path:
        """
        Get the path to the vector database, downloading if necessary.

        Args:
            auto_download: If True, automatically download if not present

        Returns:
            Path to the vector database directory

        Raises:
            FileNotFoundError: If database not found and auto_download is False
            RuntimeError: If download fails
        """
        if not self.is_downloaded():
            if auto_download:
                logger.info("Vector database not found. Downloading...")
                return self.download_and_extract()
            else:
                raise FileNotFoundError(
                    f"Vector database not found at {self.db_path}. "
                    "Set auto_download=True to download automatically."
                )

        return self.db_path

    def clear_cache(self):
        """
        Remove the downloaded vector database to free up space.

        Warning: This will require redownloading the database on next use.
        """
        if self.db_path.exists():
            import shutil
            logger.info(f"Removing cached database at: {self.db_path}")
            shutil.rmtree(self.db_path)
            logger.info("✓ Cache cleared")
        else:
            logger.info("No cache to clear")

    def get_info(self) -> dict:
        """
        Get information about the vector database.

        Returns:
            Dictionary containing database information
        """
        info = {
            "repository": VectorDBConfig.REPO_ID,
            "filename": VectorDBConfig.FILENAME,
            "extract_path": str(self.extract_path),
            "db_path": str(self.db_path),
            "is_downloaded": self.is_downloaded(),
        }

        if self.is_downloaded():
            # Calculate size
            total_size = sum(
                f.stat().st_size
                for f in self.db_path.rglob('*')
                if f.is_file()
            )
            info["size_mb"] = round(total_size / (1024 * 1024), 2)

        return info


def download_vector_db(
        extract_path: Optional[str] = None,
        force: bool = False
) -> str:
    """
    Convenience function to download the vector database.

    Args:
        extract_path: Custom extraction path
        force: Force redownload even if exists

    Returns:
        Path to the extracted database directory as string

    Example:
        >>> from shifaa.rag.vector_db import download_vector_db
        >>> db_path = download_vector_db()
        >>> print(f"Database at: {db_path}")
    """
    manager = VectorDBManager(extract_path=Path(extract_path) if extract_path else None)
    return str(manager.download_and_extract(force=force))


def download_vector_db_cli():
    """CLI entry point for downloading vector database."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Shifaa Medical Vector Database"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Custom extraction path",
        default=None
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if exists"
    )

    args = parser.parse_args()

    try:
        db_path = download_vector_db(extract_path=args.path, force=args.force)
        print(f"\n✓ Success! Vector database ready at:")
        print(f"  {db_path}")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        exit(1)