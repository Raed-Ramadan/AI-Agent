# RAG System Documentation

This project includes a Retrieval-Augmented Generation (RAG) system to enhance the Engineering Companion with document-based knowledge retrieval.

## Components

### rag_ingest.py
Handles document ingestion from various sources:
- Local files (PDF, DOCX, TXT, MD)
- Directories (bulk ingestion)
- URLs (web content)

### rag_chunking.py
Splits documents into manageable chunks for efficient retrieval:
- Recursive character splitting
- Section-based chunking
- Configurable chunk size and overlap

### rag_store.py
Manages vector storage using ChromaDB:
- Document embedding and storage
- Collection management
- Similarity search capabilities

### rag_retriever.py
Retrieves relevant documents based on queries:
- Semantic similarity search
- Metadata filtering
- Hybrid retrieval (semantic + metadata)
- Result reranking

### rag_prompting.py
Builds enhanced prompts with retrieved context:
- Context-aware prompt generation
- Conversational prompts with history
- Comparison prompts
- User context integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
- `OPENAI_API_KEY` or `OPENROUTER_API_KEY` for embeddings
- `OPENROUTER_API_KEY` for LLM generation

3. Add knowledge documents to the `knowledge/` directory

## Usage Example

```python
from rag_ingest import DocumentIngester
from rag_chunking import TextChunker
from rag_store import VectorStore
from rag_retriever import DocumentRetriever
from rag_prompting import RAGPromptBuilder

# Ingest documents
ingester = DocumentIngester()
documents = ingester.ingest_directory("knowledge")

# Chunk documents
chunker = TextChunker()
chunked_docs = chunker.chunk_documents(documents)

# Store in vector database
store = VectorStore()
store.create_collection("engineering_docs")
store.add_documents(chunked_docs)

# Set up retriever and prompt builder
retriever = DocumentRetriever(store)
prompt_builder = RAGPromptBuilder()

# Retrieve and generate response
query = "What is thermodynamics?"
retrieved_docs = retriever.retrieve(query, k=3)
prompt = prompt_builder.build_prompt(query, retrieved_docs)

# Use prompt with your LLM (OpenRouter, etc.)
```

## Integration with Main App

To integrate RAG with the main Streamlit app, modify `app.py` to:
1. Import RAG components
2. Initialize RAG system on startup
3. Use retrieved context in AI prompts
4. Add UI for document management

## File Formats Supported

- Plain text (.txt)
- Markdown (.md)
- PDF (.pdf)
- Microsoft Word (.docx, .doc)

## Configuration

- Chunk size: 1000 characters (configurable)
- Chunk overlap: 200 characters (configurable)
- Max context length: 4000 characters (configurable)
- Default embedding model: text-embedding-ada-002
- Vector store: ChromaDB (persistent)