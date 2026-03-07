"""Data loading utilities for experiments."""
import numpy as np
from typing import List, Tuple, Optional, Union
import os
import json

class DocumentCollection:
    """Simple document collection that works with user-provided docs."""
    
    def __init__(self, documents: Optional[List[str]] = None):
        """
        Initialize with either provided documents or create random ones.
        
        Args:
            documents: Optional list of document strings. If None, creates random docs.
        """
        if documents is not None:
            self.documents = documents
        else:
            self.documents = self._create_random_documents(100)
    
    def _create_random_documents(self, n: int = 100) -> List[str]:
        """Create random documents for testing."""
        topics = [
            "machine learning", "artificial intelligence", "neural networks",
            "deep learning", "reinforcement learning", "supervised learning",
            "unsupervised learning", "transformers", "attention mechanism",
            "large language models", "GPT", "BERT", "computer vision",
            "natural language processing", "speech recognition", "robotics",
            "data science", "statistics", "probability", "linear algebra"
        ]
        
        templates = [
            "{} is a field of study that deals with {}.",
            "The concept of {} is fundamental to modern {}.",
            "Researchers have made significant progress in {} recently.",
            "{} can be applied to solve problems in {}.",
            "One of the key challenges in {} is {}.",
            "The future of {} depends on advances in {}.",
            "{} has revolutionized the way we think about {}.",
            "Many companies are now adopting {} for {}.",
            "The mathematics behind {} involves {}.",
            "Understanding {} requires knowledge of {}."
        ]
        
        documents = []
        for i in range(n):
            topic1 = np.random.choice(topics)
            topic2 = np.random.choice(topics)
            template = np.random.choice(templates)
            doc = template.format(topic1, topic2)
            documents.append(doc)
        
        return documents
    
    def get_sample_queries(self, n: int = 5) -> List[str]:
        """Get sample queries for testing."""
        queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain deep learning",
            "What are transformers in AI?",
            "How does reinforcement learning differ from supervised learning?",
            "What is BERT used for?",
            "Explain attention mechanism",
            "What are large language models?",
            "How is probability used in machine learning?",
            "What is computer vision used for?"
        ]
        return queries[:n]
    
    def save(self, path: str):
        """Save documents to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.documents, f)
    
    @classmethod
    def load(cls, path: str) -> 'DocumentCollection':
        """Load documents from file."""
        with open(path, 'r') as f:
            documents = json.load(f)
        return cls(documents)
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return self.documents[idx]