"""
Azuma-Hoeffding inequality for martingale differences.
P(|M_n| ≥ ε) ≤ 2exp(-ε²/(2n c²))
where c = maximum change per step.
"""
import numpy as np

class AzumaBound:
    """
    Azuma-Hoeffding inequality for dependent sequential data.
    
    Use when documents are retrieved sequentially and may be dependent.
    This is where your martingale expertise comes in!
    """
    
    def __init__(self, confidence: float = 0.95, max_change: float = 0.2):
        """
        Args:
            confidence: Desired confidence level
            max_change: Maximum possible change when adding one document (c)
                       If unsure, set higher for conservative bounds
        """
        self.confidence = confidence
        self.delta = 1 - confidence
        self.c = max_change
        
    def epsilon(self, n: int) -> float:
        """
        Returns ε such that P(|error| ≥ ε) ≤ 1 - confidence.
        
        For the sum M_n: ε_sum = √(2n c² ln(2/δ))
        For the average: divide by n
        """
        if n == 0:
            return float('inf')
        epsilon_sum = np.sqrt(2 * n * self.c**2 * np.log(2 / self.delta))
        return epsilon_sum / n  # Convert to bound on average
    
    def lower_bound(self, scores: np.ndarray) -> float:
        """Pessimistic lower bound accounting for sequential dependence."""
        if len(scores) == 0:
            return 0.0
        mean = np.mean(scores)
        return mean - self.epsilon(len(scores))
    
    def upper_bound(self, scores: np.ndarray) -> float:
        """Optimistic upper bound."""
        if len(scores) == 0:
            return 1.0
        mean = np.mean(scores)
        return mean + self.epsilon(len(scores))
    
    def estimate_max_change(self, scores: np.ndarray) -> float:
        """
        Estimate c from data (maximum change between consecutive scores).
        Updates self.c with the estimated value.
        """
        if len(scores) < 2:
            return self.c
        
        # Calculate differences between consecutive scores
        diffs = np.abs(np.diff(scores))
        if len(diffs) > 0:
            estimated_c = max(np.max(diffs), self.c)
            self.c = estimated_c
        return self.c
    
    def __repr__(self):
        return f"AzumaBound(confidence={self.confidence}, max_change={self.c})"