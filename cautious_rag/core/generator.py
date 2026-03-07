"""Answer generation from retrieved documents."""
from typing import List, Optional

class Generator:
    """Simple generator that combines documents into a prompt."""
    
    def __init__(self, llm=None):
        """
        Initialize generator.
        
        Args:
            llm: Optional LLM client (if None, uses template-based responses)
        """
        self.llm = llm
        
    def generate(self, query: str, documents: List[str]) -> str:
        """Generate answer from query and documents."""
        if self.llm:
            # Use actual LLM if provided
            context = "\n\n".join(documents)
            prompt = f"""Based on the following documents, answer the question.

Documents:
{context}

Question: {query}

Answer:"""
            return self.llm.generate(prompt)
        else:
            # Simple template for testing
            return f"[Based on {len(documents)} documents] " + self._template_answer(query, documents)
    
    def _template_answer(self, query: str, documents: List[str]) -> str:
        """Generate a simple template answer for testing."""
        # Extract key terms from query
        words = set(query.lower().split())
        
        # Find documents with most matching words
        relevant_text = ""
        max_matches = 0
        
        for doc in documents:
            doc_words = set(doc.lower().split())
            matches = len(words & doc_words)
            if matches > max_matches:
                max_matches = matches
                relevant_text = doc[:100] + "..."
        
        if relevant_text:
            return f"The most relevant information I found: {relevant_text}"
        else:
            return "I found some documents but couldn't extract a specific answer."