"""Document retrieval functionality."""
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

class Retriever:
    """Retriever that finds relevant documents for a query."""
    
    def __init__(self, documents: List[str], model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize retriever with documents.
        
        Args:
            documents: List of document texts
            model_name: Sentence transformer model for embeddings
        """
        self.documents = documents
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print(f"Encoding {len(documents)} documents...")
        self.embeddings = self.model.encode(documents, show_progress_bar=True)
        
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Return top-k document texts."""
        docs_with_scores = self.retrieve_with_scores(query, k)
        return [doc for doc, _ in docs_with_scores]
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k documents with relevance scores."""
        query_emb = self.model.encode([query])[0]
        
        # Cosine similarity
        scores = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Get top k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # Normalize scores to [0, 1] range for concentration bounds
        min_score = scores[top_indices].min()
        max_score = scores[top_indices].max()
        if max_score > min_score:
            normalized_scores = (scores[top_indices] - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones(k) * 0.5
            
        return [(self.documents[i], normalized_scores[j]) 
                for j, i in enumerate(top_indices)]