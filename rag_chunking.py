# rag_chunking.py
# Handles text chunking for RAG system

from typing import List, Dict, Any
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create a configured text splitter for the chunker."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """Chunk a list of documents into smaller pieces."""
        langchain_docs = []

        for doc in documents:
            # Convert to LangChain Document format
            lc_doc = Document(
                page_content=doc['content'],
                metadata=doc.get('metadata', {})
            )
            langchain_docs.append(lc_doc)

        text_splitter = self._get_text_splitter()
        chunked_docs = text_splitter.split_documents(langchain_docs)
        return chunked_docs

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Chunk a single text string."""
        if metadata is None:
            metadata = {}

        doc = Document(page_content=text, metadata=metadata)
        text_splitter = self._get_text_splitter()

        return text_splitter.split_documents([doc])

    def chunk_by_sections(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Chunk text by identifying sections (headers, paragraphs)."""
        if metadata is None:
            metadata = {}

        # Split by double newlines (paragraphs) and headers
        sections = re.split(r'(\n#{1,6}.*?\n|\n\n)', text)

        chunks = []
        current_chunk = ""
        current_metadata = metadata.copy()

        for section in sections:
            if len(current_chunk + section) > self.chunk_size and current_chunk:
                # Create chunk
                doc = Document(
                    page_content=current_chunk.strip(),
                    metadata=current_metadata
                )
                chunks.append(doc)

                # Start new chunk with overlap
                overlap_size = min(self.chunk_overlap, len(current_chunk))
                current_chunk = current_chunk[-overlap_size:] + section
            else:
                current_chunk += section

        # Add remaining chunk
        if current_chunk.strip():
            doc = Document(
                page_content=current_chunk.strip(),
                metadata=current_metadata
            )
            chunks.append(doc)

        return chunks

# Usage example
if __name__ == "__main__":
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    sample_text = "This is a long document.\n\nIt has multiple paragraphs.\n\n## Section Header\n\nMore content here."
    chunks = chunker.chunk_text(sample_text)
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {len(chunk.page_content)} characters")