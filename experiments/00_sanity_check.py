#!/usr/bin/env python
"""Simple test to verify everything works."""

import sys
import numpy as np
sys.path.append('..')

from cautious_rag.bounds.hoeffding import HoeffdingBound
from cautious_rag.bounds.azuma import AzumaBound
from cautious_rag.bounds.bernstein import EmpiricalBernsteinBound
from cautious_rag.decision.engine import DecisionEngine, BoundType

def test_bounds():
    """Test all concentration bounds."""
    print("=" * 50)
    print("Testing Concentration Bounds")
    print("=" * 50)
    
    # Generate some sample scores
    np.random.seed(42)
    scores = np.random.beta(2, 2, 20)  # Beta distribution, mean ~0.5
    
    print(f"\nSample scores (n={len(scores)}):")
    print(f"Mean: {np.mean(scores):.3f}")
    print(f"Std:  {np.std(scores):.3f}")
    print()
    
    # Test Hoeffding
    hoeffding = HoeffdingBound(confidence=0.95)
    lower_h = hoeffding.lower_bound(scores)
    print(f"Hoeffding: lower bound = {lower_h:.4f}")
    
    # Test Azuma
    azuma = AzumaBound(confidence=0.95, max_change=0.2)
    azuma.estimate_max_change(scores)
    lower_a = azuma.lower_bound(scores)
    print(f"Azuma:     lower bound = {lower_a:.4f} (c={azuma.c:.3f})")
    
    # Test Bernstein
    bernstein = EmpiricalBernsteinBound(confidence=0.95)
    lower_b = bernstein.lower_bound(scores)
    print(f"Bernstein: lower bound = {lower_b:.4f}")
    
    print("\n" + "=" * 50)
    
    return hoeffding, azuma, bernstein


def test_decision_engine():
    """Test the decision engine."""
    print("\n" + "=" * 50)
    print("Testing Decision Engine")
    print("=" * 50)
    
    engine = DecisionEngine(
    threshold=0.6,        
    confidence=0.95,
    default_bound=BoundType.ADAPTIVE,
    verbose=True
    )
    
    # Test with different numbers of documents
    np.random.seed(123)
    
    test_cases = [
        ("Good docs", np.random.beta(5, 2, 10)),   # High relevance
        ("Mixed docs", np.random.beta(2, 2, 10)),  # Medium relevance
        ("Poor docs", np.random.beta(2, 5, 10)),   # Low relevance
        ("Few docs", np.random.beta(3, 3, 3)),     # Small sample
    ]
    
    for name, scores in test_cases:
        print(f"\n--- {name} ---")
        print(f"Scores: mean={np.mean(scores):.3f}, n={len(scores)}")
        
        decision = engine.decide(scores)
        print(f"Decision: {decision.message}")
        print(f"Lower bound: {decision.lower_bound:.4f} ({decision.bound_type})")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Decision Summary")
    print("=" * 50)
    summary = engine.summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return engine


if __name__ == "__main__":
    test_bounds()
    test_decision_engine()
    print("\n✅ All tests complete!")

