"""Decision engine: when to answer, when to be cautious."""
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from ..bounds.hoeffding import HoeffdingBound
from ..bounds.azuma import AzumaBound
from ..bounds.bernstein import EmpiricalBernsteinBound


class BoundType(Enum):
    """Types of concentration bounds available."""
    HOEFFDING = "hoeffding"
    AZUMA = "azuma"
    BERNSTEIN = "bernstein"
    ADAPTIVE = "adaptive"


@dataclass
class Decision:
    """Result of a decision: whether to answer and why."""
    can_answer: bool
    lower_bound: float
    bound_type: str
    confidence: float
    n_docs: int
    message: str
    answer: Optional[str] = None
    metadata: Optional[Dict] = None


class DecisionEngine:
    """
    Decides whether to answer based on concentration bounds.
    
    The engine retrieves documents and applies concentration inequalities
    to get a pessimistic lower bound on true relevance. If this bound
    exceeds a threshold, it answers; otherwise, it refuses.
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        confidence: float = 0.95,
        default_bound: BoundType = BoundType.ADAPTIVE,
        max_change: float = 0.2,
        verbose: bool = False
    ):
        """
        Args:
            threshold: Minimum guaranteed relevance to answer
            confidence: Statistical confidence level
            default_bound: Which bound to use by default
            max_change: Max change per step for Azuma (if needed)
            verbose: Print debug information
        """
        self.threshold = threshold
        self.confidence = confidence
        self.default_bound = default_bound
        self.verbose = verbose
        self.max_change = max_change
        
        # Initialize bounds
        self.hoeffding = HoeffdingBound(confidence)
        self.azuma = AzumaBound(confidence, max_change)
        self.bernstein = EmpiricalBernsteinBound(confidence)
        
        # Track history
        self.decisions = []
        
    def decide(
        self,
        scores: List[float],
        bound_type: Optional[BoundType] = None,
        query: str = "",
        generator=None
    ) -> Decision:
        """
        Decide whether to answer based on relevance scores.
        
        Args:
            scores: Relevance scores for retrieved documents
            bound_type: Which bound to use (None = use default)
            query: Original query (for generating answer)
            generator: Generator to produce answer
            
        Returns:
            Decision object
        """
        scores = np.array(scores)
        n = len(scores)
        
        if n == 0:
            return Decision(
                can_answer=False,
                lower_bound=0.0,
                bound_type="none",
                confidence=self.confidence,
                n_docs=0,
                message="No documents retrieved"
            )
        
        # Choose bound
        bound_type = bound_type or self.default_bound
        
        if bound_type == BoundType.HOEFFDING:
            lower = self.hoeffding.lower_bound(scores)
            bound_name = "hoeffding"
        elif bound_type == BoundType.AZUMA:
            # Estimate max change from data
            self.azuma.estimate_max_change(scores)
            lower = self.azuma.lower_bound(scores)
            bound_name = f"azuma(c={self.azuma.c:.3f})"
        elif bound_type == BoundType.BERNSTEIN:
            lower = self.bernstein.lower_bound(scores)
            bound_name = "bernstein"
        else:  # ADAPTIVE - use tightest bound
            bounds = {
                "hoeffding": self.hoeffding.lower_bound(scores),
                "azuma": self.azuma.lower_bound(scores),
            }
            if n >= 5:
                bounds["bernstein"] = self.bernstein.lower_bound(scores)
            
            # Tightest bound = highest lower bound
            bound_name = max(bounds, key=bounds.get)
            lower = bounds[bound_name]
        
        # Decision
        can_answer = lower > self.threshold
        
        # Generate message
        if can_answer:
            message = f"✅ Answering (lower bound {lower:.3f} > {self.threshold})"
        else:
            # How many more docs might be needed?
            gap = self.threshold - lower
            if gap > 0:
                # Rough estimate using Hoeffding
                n_needed = self.hoeffding.sample_size_needed(gap)
                message = f"❌ Need ≈{n_needed} more docs (gap={gap:.3f})"
            else:
                message = f"❌ Cannot answer (lower bound {lower:.3f} ≤ {self.threshold})"
        
        # Generate answer if requested
        answer = None
        if can_answer and generator and query:
            answer = generator.generate(query, [])  # You'd pass actual docs
        
        if self.verbose:
            print(f"n={n}, mean={np.mean(scores):.3f}, lower={lower:.3f}, {bound_name}")
        
        decision = Decision(
            can_answer=can_answer,
            lower_bound=lower,
            bound_type=bound_name,
            confidence=self.confidence,
            n_docs=n,
            message=message,
            answer=answer,
            metadata={
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "scores": scores.tolist()
            }
        )
        
        self.decisions.append(decision)
        return decision
    
    def summary(self) -> Dict:
        """Get summary of all decisions made."""
        if not self.decisions:
            return {}
        
        answered = [d.can_answer for d in self.decisions]
        
        return {
            "total_queries": len(self.decisions),
            "answered_rate": sum(answered) / len(answered),
            "refusal_rate": 1 - sum(answered) / len(answered),
            "avg_lower_bound": np.mean([d.lower_bound for d in self.decisions]),
            "avg_docs": np.mean([d.n_docs for d in self.decisions]),
            "bounds_used": {
                name: sum(1 for d in self.decisions if d.bound_type.startswith(name))
                for name in set(d.bound_type.split('(')[0] for d in self.decisions)
            }
        }