# rag_store.py
# Handles vector storage for RAG system

import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class VectorStore:
    def __init__(self, persist_directory: str = "vectorstore", embedding_model: str = "text-embedding-ada-002", base_url: str = None, api_key: str = None):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided either as parameter or environment variable (OPENAI_API_KEY or OPENROUTER_API_KEY)")

        # Initialize embeddings
        embeddings_kwargs = {
            "model": embedding_model,
            "openai_api_key": api_key
        }
        if base_url:
            embeddings_kwargs["openai_api_base"] = base_url
        self.embeddings = OpenAIEmbeddings(**embeddings_kwargs)

        # Initialize vector store
        self.vectorstore = None
        self.load_or_create()

    def load_or_create(self):
        """Load existing FAISS index or create new one."""
        index_path = os.path.join(self.persist_directory, "faiss_index")
        if os.path.exists(index_path):
            try:
                self.vectorstore = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            except:
                self.vectorstore = None
        if self.vectorstore is None:
            # Create empty FAISS index
            self.vectorstore = FAISS.from_texts(["dummy"], self.embeddings)
            self.save()

    def save(self):
        """Save the FAISS index to disk."""
        if self.vectorstore:
            index_path = os.path.join(self.persist_directory, "faiss_index")
            os.makedirs(index_path, exist_ok=True)
            self.vectorstore.save_local(index_path)

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        if self.vectorstore:
            self.vectorstore.add_documents(documents)
            self.save()

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        if self.vectorstore:
            return self.vectorstore.similarity_search(query, k=k)
        return []

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Perform similarity search with relevance scores."""
        if self.vectorstore:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        return []

# Usage example
if __name__ == "__main__":
    store = VectorStore()

    # Create collection
    store.create_collection("test_collection")

    # Add some sample documents
    from langchain_core.documents import Document
    docs = [
        Document(page_content="This is about mechanical engineering.", metadata={"subject": "mechanical"}),
        Document(page_content="Electrical engineering fundamentals.", metadata={"subject": "electrical"})
    ]

    store.add_documents(docs)

    # Search
    results = store.similarity_search("mechanical engineering")
    print(f"Found {len(results)} results")
    for result in results:
        print(result.page_content)