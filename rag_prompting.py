# rag_prompting.py
# Handles prompt engineering for RAG system

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

class RAGPromptBuilder:
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length

    def build_prompt(self, query: str, retrieved_docs: List[Document],
                    system_prompt: str = None, user_context: Dict[str, Any] = None) -> str:
        """Build a RAG-enhanced prompt."""
        # Extract relevant context from retrieved documents
        context = self._extract_context(retrieved_docs)

        # Build the prompt
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        prompt_parts = [system_prompt]

        if user_context:
            prompt_parts.append(self._format_user_context(user_context))

        prompt_parts.append(f"Context:\n{context}")
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("Answer:")

        return "\n\n".join(prompt_parts)

    def build_conversational_prompt(self, query: str, retrieved_docs: List[Document],
                                   conversation_history: List[Dict[str, str]],
                                   system_prompt: str = None) -> str:
        """Build a conversational RAG prompt with chat history."""
        context = self._extract_context(retrieved_docs)

        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        prompt_parts = [system_prompt]

        # Add conversation history
        if conversation_history:
            history_text = self._format_conversation_history(conversation_history)
            prompt_parts.append(f"Conversation History:\n{history_text}")

        prompt_parts.append(f"Retrieved Context:\n{context}")
        prompt_parts.append(f"Current Question: {query}")
        prompt_parts.append("Response:")

        return "\n\n".join(prompt_parts)

    def _extract_context(self, documents: List[Document]) -> str:
        """Extract and format context from retrieved documents."""
        context_parts = []
        total_length = 0

        for doc in documents:
            content = doc.page_content.strip()
            metadata = doc.metadata

            # Format document with metadata
            doc_text = content
            if metadata:
                source_info = []
                if 'source' in metadata:
                    source_info.append(f"Source: {metadata['source']}")
                if 'filename' in metadata:
                    source_info.append(f"File: {metadata['filename']}")
                if source_info:
                    doc_text = f"{' | '.join(source_info)}\n{content}"

            # Check if adding this document would exceed max length
            if total_length + len(doc_text) > self.max_context_length:
                # Truncate if necessary
                remaining_length = self.max_context_length - total_length
                if remaining_length > 100:  # Only add if we have space for meaningful content
                    doc_text = doc_text[:remaining_length] + "..."
                    context_parts.append(doc_text)
                break

            context_parts.append(doc_text)
            total_length += len(doc_text)

        return "\n\n---\n\n".join(context_parts)

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for engineering education."""
        return """You are an expert engineering instructor and AI assistant. Your role is to help students learn engineering concepts through clear, accurate, and engaging explanations.

Guidelines:
- Use the provided context to inform your answers
- Explain concepts step-by-step when appropriate
- Include relevant examples and applications
- Be encouraging and supportive of student learning
- If the context doesn't fully answer the question, use your general knowledge but note when you're doing so
- Maintain accuracy in technical explanations
- Ask clarifying questions when the query is ambiguous"""

    def _format_user_context(self, user_context: Dict[str, Any]) -> str:
        """Format user context information."""
        context_parts = []

        if 'subject' in user_context:
            context_parts.append(f"Subject: {user_context['subject']}")

        if 'level' in user_context:
            context_parts.append(f"Student Level: {user_context['level']}")

        if 'learning_goals' in user_context:
            context_parts.append(f"Learning Goals: {user_context['learning_goals']}")

        if 'previous_topics' in user_context:
            context_parts.append(f"Previously Covered: {', '.join(user_context['previous_topics'])}")

        return "User Context:\n" + "\n".join(context_parts)

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for the prompt."""
        formatted_history = []

        for entry in history[-5:]:  # Keep last 5 exchanges
            if 'user' in entry and 'assistant' in entry:
                formatted_history.append(f"User: {entry['user']}")
                formatted_history.append(f"Assistant: {entry['assistant']}")

        return "\n".join(formatted_history)

    def build_comparison_prompt(self, query: str, retrieved_docs: List[Document],
                               comparison_criteria: List[str]) -> str:
        """Build a prompt for comparing different approaches or concepts."""
        context = self._extract_context(retrieved_docs)

        system_prompt = """You are an expert engineering educator specializing in comparative analysis.
Your task is to provide balanced, well-structured comparisons of engineering concepts, methods, or technologies."""

        prompt_parts = [system_prompt]

        prompt_parts.append(f"Context:\n{context}")
        prompt_parts.append(f"Comparison Query: {query}")
        prompt_parts.append(f"Comparison Criteria: {', '.join(comparison_criteria)}")
        prompt_parts.append("Provide a structured comparison addressing each criterion:")

        return "\n\n".join(prompt_parts)

# Usage example
if __name__ == "__main__":
    from langchain_core.documents import Document

    # Sample documents
    docs = [
        Document(page_content="Mechanical engineering involves the design and analysis of mechanical systems.",
                metadata={"subject": "mechanical"}),
        Document(page_content="Thermodynamics is a branch of physics dealing with heat and energy.",
                metadata={"subject": "thermodynamics"})
    ]

    # Build prompt
    builder = RAGPromptBuilder()
    prompt = builder.build_prompt(
        query="What is mechanical engineering?",
        retrieved_docs=docs,
        user_context={"level": "beginner", "subject": "mechanical"}
    )

    print("Generated Prompt:")
    print(prompt)