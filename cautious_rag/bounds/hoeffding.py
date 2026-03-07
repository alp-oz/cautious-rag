"""Hoeffding's inequality for independent bounded random variables."""
import numpy as np

class HoeffdingBound:
    """
    Hoeffding's inequality: P(|sample_mean - true_mean| ≥ ε) ≤ 2exp(-2nε²)
    
    Use when documents are retrieved independently.
    """
    
    def __init__(self, confidence: float = 0.95):
        """
        Args:
            confidence: Desired confidence level (e.g., 0.95 for 95%)
        """
        self.confidence = confidence
        self.delta = 1 - confidence
        
    def epsilon(self, n: int) -> float:
        """
        Returns ε such that P(|error| ≥ ε) ≤ 1 - confidence.
        
        Derived from: 2exp(-2nε²) = delta
        => exp(-2nε²) = delta/2
        => -2nε² = ln(delta/2)
        => ε² = -ln(delta/2)/(2n)
        => ε = √(-ln(delta/2)/(2n))
        """
        if n == 0:
            return float('inf')
        return np.sqrt(-np.log(self.delta / 2) / (2 * n))
    
    def lower_bound(self, scores: np.ndarray) -> float:
        """
        Pessimistic lower bound on true mean with given confidence.
        
        Returns L such that P(true_mean ≥ L) ≥ confidence.
        """
        if len(scores) == 0:
            return 0.0
        mean = np.mean(scores)
        return mean - self.epsilon(len(scores))
    
    def upper_bound(self, scores: np.ndarray) -> float:
        """Optimistic upper bound on true mean."""
        if len(scores) == 0:
            return 1.0
        mean = np.mean(scores)
        return mean + self.epsilon(len(scores))
    
    def confidence_interval(self, scores: np.ndarray) -> tuple:
        """Return (lower, upper) confidence interval."""
        return (self.lower_bound(scores), self.upper_bound(scores))
    
    def sample_size_needed(self, margin: float) -> int:
        """
        How many samples needed to achieve given margin with desired confidence.
        
        Solve: ε = margin => √(-ln(δ/2)/(2n)) = margin
        => -ln(δ/2)/(2n) = margin²
        => n = -ln(δ/2)/(2 * margin²)
        """
        if margin <= 0:
            return 0
        return int(np.ceil(-np.log(self.delta / 2) / (2 * margin**2)))
    
    def __repr__(self):
        return f"HoeffdingBound(confidence={self.confidence})"