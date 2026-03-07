"""Evaluation metrics for RAG systems."""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class EvaluationResult:
    """Results from evaluating a RAG system."""
    
    # Basic stats
    total_queries: int = 0
    answered_queries: int = 0
    refused_queries: int = 0
    
    # Quality metrics
    avg_relevance: float = 0.0
    avg_lower_bound: float = 0.0
    avg_docs_retrieved: float = 0.0
    
    # Bound usage
    bound_counts: Dict[str, int] = field(default_factory=dict)
    
    # Per-query details
    details: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def answer_rate(self) -> float:
        """Percentage of queries answered."""
        if self.total_queries == 0:
            return 0.0
        return self.answered_queries / self.total_queries
    
    @property
    def refusal_rate(self) -> float:
        """Percentage of queries refused."""
        return 1 - self.answer_rate
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total queries:       {self.total_queries}")
        print(f"Answered:            {self.answered_queries} ({self.answer_rate*100:.1f}%)")
        print(f"Refused:             {self.refused_queries} ({self.refusal_rate*100:.1f}%)")
        print(f"Avg relevance:       {self.avg_relevance:.3f}")
        print(f"Avg lower bound:     {self.avg_lower_bound:.3f}")
        print(f"Avg docs retrieved:  {self.avg_docs_retrieved:.1f}")
        print("\nBound usage:")
        for bound, count in self.bound_counts.items():
            pct = 100 * count / self.total_queries
            print(f"  {bound}: {count} ({pct:.1f}%)")
        print("=" * 50)


class MetricsCalculator:
    """Calculate metrics for RAG evaluation."""
    
    @staticmethod
    def from_decisions(decisions: List[Any]) -> EvaluationResult:
        """
        Create evaluation result from list of Decision objects.
        
        Args:
            decisions: List of Decision objects from DecisionEngine
        """
        result = EvaluationResult()
        
        for d in decisions:
            result.total_queries += 1
            
            # Track answer/refusal - Decision objects use .can_answer, not .get()
            if d.can_answer:
                result.answered_queries += 1
            else:
                result.refused_queries += 1
            
            # Accumulate metrics
            if d.metadata and 'mean' in d.metadata:
                result.avg_relevance += d.metadata['mean']
            result.avg_lower_bound += d.lower_bound
            result.avg_docs_retrieved += d.n_docs
            
            # Track bound type
            bound_type = d.bound_type.split('(')[0] if '(' in d.bound_type else d.bound_type
            result.bound_counts[bound_type] = result.bound_counts.get(bound_type, 0) + 1
            
            # Store detail (convert to dict for consistency)
            result.details.append({
                'can_answer': d.can_answer,
                'lower_bound': d.lower_bound,
                'bound_type': d.bound_type,
                'n_docs': d.n_docs,
                'message': d.message,
                'metadata': d.metadata
            })
        
        # Average
        if result.total_queries > 0:
            result.avg_relevance /= result.total_queries
            result.avg_lower_bound /= result.total_queries
            result.avg_docs_retrieved /= result.total_queries
        
        return result
    
    @staticmethod
    def compare_bounds(results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """
        Compare performance of different bound types.
        
        Args:
            results: Dict mapping bound names to EvaluationResult
        """
        comparison = {}
        
        for name, result in results.items():
            comparison[name] = {
                'answer_rate': result.answer_rate,
                'avg_lower_bound': result.avg_lower_bound,
                'avg_docs': result.avg_docs_retrieved,
                'refusal_rate': result.refusal_rate
            }
        
        return comparison
    
    @staticmethod
    def plot_ready_data(decisions: List[Any]) -> Dict[str, np.ndarray]:
        """
        Prepare data for visualization from Decision objects.
        
        Returns dict with arrays for plotting.
        """
        n_docs = np.array([d.n_docs for d in decisions])
        lower_bounds = np.array([d.lower_bound for d in decisions])
        means = np.array([d.metadata.get('mean', 0) if d.metadata else 0 for d in decisions])
        answered = np.array([d.can_answer for d in decisions])
        
        return {
            'n_docs': n_docs,
            'lower_bounds': lower_bounds,
            'means': means,
            'answered': answered
        }