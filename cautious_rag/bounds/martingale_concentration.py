"""
Martingale-based concentration bounds for dependent sequences.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple

class MartingaleConcentration:
    """
    Concentration bounds for dependent sequences using martingales.
    
    Uses Azuma-Hoeffding inequality for martingale differences:
    P(|S_n - E[S_n]| ≥ ε) ≤ 2exp(-ε²/(2n c²))
    where c = maximum change per step.
    """
    
    def __init__(self, confidence: float = 0.95):
        """
        Args:
            confidence: Desired confidence level (e.g., 0.95 for 95%)
        """
        self.confidence = confidence
        self.delta = 1 - confidence
        
    def azuma_hoeffding_bound(self, scores: List[float], max_change: Optional[float] = None) -> Dict:
        """
        Apply Azuma-Hoeffding bound to sequence of scores.
        
        Args:
            scores: List of relevance scores (assumed to be a martingale difference sequence)
            max_change: Maximum change per step (c). If None, estimated from data.
            
        Returns:
            Dict with lower_bound, upper_bound, epsilon, max_change, method
        """
        n = len(scores)
        if n == 0:
            return {
                'lower_bound': 0.0,
                'upper_bound': 1.0,
                'epsilon': float('inf'),
                'max_change': 0.0,
                'method': 'azuma-hoeffding'
            }
        
        mean = np.mean(scores)
        
        # Estimate max change if not provided
        if max_change is None:
            if n > 1:
                # Maximum difference between consecutive scores
                diffs = np.abs(np.diff(scores))
                max_change = np.max(diffs) if len(diffs) > 0 else 0.2
            else:
                max_change = 0.2  # Default
        
        # Azuma-Hoeffding for the average
        # P(|avg - μ| ≥ ε) ≤ 2exp(-n ε²/(2 c²))
        epsilon = np.sqrt(2 * max_change**2 * np.log(2 / self.delta) / n)
        
        return {
            'lower_bound': mean - epsilon,
            'upper_bound': mean + epsilon,
            'epsilon': epsilon,
            'max_change': max_change,
            'method': 'azuma-hoeffding'
        }
    
    def compare_with_hoeffding(self, scores: List[float]) -> Dict:
        """
        Compare martingale bound with Hoeffding (independence assumption).
        """
        n = len(scores)
        mean = np.mean(scores)
        
        # Hoeffding (assumes independence)
        hoeffding_eps = np.sqrt(np.log(2 / self.delta) / (2 * n))
        hoeffding_lower = mean - hoeffding_eps
        
        # Azuma (handles dependence)
        azuma_result = self.azuma_hoeffding_bound(scores)
        
        return {
            'n': n,
            'mean': mean,
            'hoeffding': {
                'lower_bound': hoeffding_lower,
                'epsilon': hoeffding_eps
            },
            'azuma': azuma_result,
            'difference': azuma_result['lower_bound'] - hoeffding_lower
        }


class AdaptiveMartingaleConcentration(MartingaleConcentration):
    """
    Martingale bounds with adaptive max change estimation.
    """
    
    def estimate_max_change_adaptive(self, scores: List[float], window: int = 5) -> float:
        """
        Estimate max change using rolling windows.
        
        Args:
            scores: List of relevance scores
            window: Size of rolling window for volatility estimation
            
        Returns:
            Estimated max change (c)
        """
        if len(scores) < 2:
            return 0.2
        
        # Method 1: Maximum consecutive difference
        max_consecutive = np.max(np.abs(np.diff(scores)))
        
        # Method 2: Rolling window volatility
        rolling_max_changes = []
        for i in range(len(scores) - window):
            window_scores = scores[i:i+window]
            if len(window_scores) > 1:
                window_changes = np.abs(np.diff(window_scores))
                rolling_max_changes.append(np.max(window_changes))
        
        if rolling_max_changes:
            rolling_estimate = np.mean(rolling_max_changes) + np.std(rolling_max_changes)
        else:
            rolling_estimate = max_consecutive
        
        # Use the more conservative estimate with a minimum
        c = max(max_consecutive, rolling_estimate, 0.05)
        
        return c
    
    def bound_with_adaptive_c(self, scores: List[float]) -> Dict:
        """
        Get bound using adaptively estimated max change.
        """
        c = self.estimate_max_change_adaptive(scores)
        return self.azuma_hoeffding_bound(scores, max_change=c)
    
    def estimate_dependence_strength(self, scores: List[float]) -> Dict:
        """
        Estimate how strong the dependence is in the sequence.
        
        Returns:
            Dict with dependence metrics
        """
        if len(scores) < 3:
            return {'dependence': 'unknown', 'reason': 'too few samples'}
        
        # Autocorrelation at lag 1
        autocorr = np.corrcoef(scores[:-1], scores[1:])[0, 1]
        
        # Variance ratio compared to independent case
        independent_var = np.var(scores) / len(scores)
        actual_var = np.var(np.cumsum(scores)) / len(scores)**2  # Rough estimate
        
        var_ratio = actual_var / independent_var if independent_var > 0 else 1.0
        
        # Interpretation
        if abs(autocorr) < 0.1:
            dependence = 'weak'
        elif autocorr > 0.5:
            dependence = 'strong positive'
        elif autocorr < -0.3:
            dependence = 'negative'
        else:
            dependence = 'moderate'
        
        return {
            'autocorrelation': autocorr,
            'variance_ratio': var_ratio,
            'dependence_strength': dependence,
            'recommendation': 'Use Azuma' if abs(autocorr) > 0.2 else 'Hoeffding may suffice'
        }