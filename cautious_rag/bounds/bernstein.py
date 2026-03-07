"""
Bernstein and Empirical Bernstein bounds.
Use variance information for tighter bounds when possible.
"""
import numpy as np

class EmpiricalBernsteinBound:
    """
    Empirical Bernstein bound: uses sample variance for tighter bounds.
    
    P(|mean - μ| ≥ ε) ≤ 2exp(-nε²/(2σ² + 2Rε/3))
    where σ² is variance and R is range.
    """
    
    def __init__(self, confidence: float = 0.95, max_range: float = 1.0):
        self.confidence = confidence
        self.delta = 1 - confidence
        self.R = max_range  # Range of values (0 to 1 for relevance)
        
    def epsilon(self, scores: np.ndarray) -> float:
        """
        Compute ε using empirical Bernstein.
        This solves for ε in: 2exp(-nε²/(2σ̂² + 2Rε/3)) = δ
        """
        n = len(scores)
        if n < 2:
            return float('inf')
        
        mean = np.mean(scores)
        variance = np.var(scores, ddof=1)  # Sample variance
        
        if variance == 0:
            # If all scores identical, use Hoeffding
            return np.sqrt(-np.log(self.delta / 2) / (2 * n))
        
        # Empirical Bernstein bound (simplified version)
        # ε ≤ √(2v log(2/δ)/n) + (R log(2/δ))/(3n)
        log_term = np.log(2 / self.delta)
        var_term = np.sqrt(2 * variance * log_term / n)
        range_term = (self.R * log_term) / (3 * n)
        
        return var_term + range_term
    
    def lower_bound(self, scores: np.ndarray) -> float:
        """Pessimistic lower bound using empirical Bernstein."""
        if len(scores) < 2:
            return 0.0
        mean = np.mean(scores)
        return mean - self.epsilon(scores)
    
    def __repr__(self):
        return f"EmpiricalBernsteinBound(confidence={self.confidence})"