# rag_retriever.py
# Handles document retrieval for RAG system

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from rag_store import VectorStore

class DocumentRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5, score_threshold: float = None) -> List[Document]:
        """Retrieve relevant documents for a query."""
        if score_threshold is not None:
            # Use similarity search with scores and filter by threshold
            results_with_scores = self.vector_store.similarity_search_with_score(query, k=k*2)  # Get more to filter

            filtered_results = []
            for doc, score in results_with_scores:
                if score >= score_threshold:
                    filtered_results.append(doc)
                    if len(filtered_results) >= k:
                        break

            return filtered_results
        else:
            return self.vector_store.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve documents with relevance scores."""
        return self.vector_store.similarity_search_with_score(query, k=k)

    def retrieve_by_metadata(self, metadata_filter: Dict[str, Any], k: int = 10) -> List[Document]:
        """Retrieve documents by metadata filter."""
        # This is a simplified implementation
        # In a real system, you'd use vector store's metadata filtering capabilities
        all_docs = self.vector_store.similarity_search("", k=1000)  # Get many docs

        filtered_docs = []
        for doc in all_docs:
            if self._matches_metadata(doc.metadata, metadata_filter):
                filtered_docs.append(doc)
                if len(filtered_docs) >= k:
                    break

        return filtered_docs

    def _matches_metadata(self, doc_metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if document metadata matches the filter."""
        for key, value in filter_metadata.items():
            if key not in doc_metadata or doc_metadata[key] != value:
                return False
        return True

    def hybrid_retrieve(self, query: str, metadata_filter: Dict[str, Any] = None,
                       k: int = 5, score_threshold: float = None) -> List[Document]:
        """Perform hybrid retrieval combining semantic search and metadata filtering."""
        # First, get candidates by semantic search
        candidates = self.retrieve(query, k=k*3, score_threshold=score_threshold)

        # Then filter by metadata if provided
        if metadata_filter:
            filtered_candidates = [
                doc for doc in candidates
                if self._matches_metadata(doc.metadata, metadata_filter)
            ]
            return filtered_candidates[:k]
        else:
            return candidates[:k]

    def rerank_results(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank retrieved documents based on relevance to query."""
        # Simple reranking based on query term frequency
        # In a real system, you'd use a more sophisticated reranking model
        query_terms = set(query.lower().split())

        scored_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            score = sum(1 for term in query_terms if term in content_lower)
            scored_docs.append((doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs[:top_k]]

# Usage example
if __name__ == "__main__":
    from rag_store import VectorStore

    # Initialize store and retriever
    store = VectorStore()
    retriever = DocumentRetriever(store)

    # Load existing collection
    if store.load_collection("test_collection"):
        # Retrieve documents
        results = retriever.retrieve("engineering", k=3)
        print(f"Retrieved {len(results)} documents")
        for result in results:
            print(f"- {result.page_content[:100]}...")
    else:
        print("No collection found. Run rag_store.py first to create a collection.")