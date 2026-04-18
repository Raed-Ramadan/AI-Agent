# rag_ingest.py
# Handles document ingestion for RAG system

import os
import requests
from typing import List, Dict, Any
from pathlib import Path
from docx import Document


class DocumentIngester:
    def __init__(self, data_dir: str = "knowledge"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def ingest_file(self, file_path: str) -> str:
        """Ingest a single file and return its content."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = ""
        if file_path.suffix.lower() == '.txt':
            content = self._read_text_file(file_path)
        elif file_path.suffix.lower() == '.md':
            content = self._read_markdown_file(file_path)
        elif file_path.suffix.lower() == '.pdf':
            content = self._read_pdf_file(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            content = self._read_docx_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        return content

    def ingest_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Ingest all supported files from a directory."""
        directory = Path(directory)
        documents = []

        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md', '.pdf', '.docx', '.doc']:
                try:
                    content = self.ingest_file(str(file_path))
                    documents.append({
                        'content': content,
                        'metadata': {
                            'source': str(file_path),
                            'filename': file_path.name,
                            'file_type': file_path.suffix.lower()
                        }
                    })
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")

        return documents

    def ingest_url(self, url: str) -> str:
        """Ingest content from a URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            raise Exception(f"Error fetching URL {url}: {e}")

    def _read_text_file(self, file_path: Path) -> str:
        """Read plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_markdown_file(self, file_path: Path) -> str:
        """Read markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_pdf_file(self, file_path: Path) -> str:
        """Read PDF file."""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 is required to read PDF files. Please install it with: pip install PyPDF2")
        
        content = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        return content

    def _read_docx_file(self, file_path: Path) -> str:
        """Read Word document."""
        doc = Document(file_path)
        content = ""
        for paragraph in doc.paragraphs:
            content += paragraph.text + "\n"
        return content

# Usage example
if __name__ == "__main__":
    ingester = DocumentIngester()
    documents = ingester.ingest_directory("knowledge")
    print(f"Ingested {len(documents)} documents")