#!/usr/bin/env python
"""
Experiment 1: Compare different concentration bounds.
Run with: python experiments/01_compare_bounds.py
"""
import sys
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path so we can import cautious_rag
sys.path.append(str(Path(__file__).parent.parent))

from cautious_rag.utils.data import DocumentCollection
from cautious_rag.core.retriever import Retriever
from cautious_rag.core.generator import Generator
from cautious_rag.decision.engine import DecisionEngine, BoundType
from cautious_rag.utils.metrics import MetricsCalculator


def make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    else:
        return obj


def run_comparison():
    """Run comparison experiment."""
    print("=" * 60)
    print("EXPERIMENT 1: Comparing Concentration Bounds")
    print("=" * 60)
    
    # 1. Create test data
    print("\n📚 Creating test documents...")
    docs = DocumentCollection()  # Creates random documents
    queries = docs.get_sample_queries(100)  # instead of 20    
    # 2. Initialize components
    print("🔧 Initializing retriever...")
    retriever = Retriever(docs.documents)  # Uses sentence-transformers
    
    print("🤖 Initializing generator (mock)...")
    generator = Generator()  # Simple template-based
    
    # 3. Test each bound type
    bound_types = [
        BoundType.HOEFFDING,
        BoundType.AZUMA,
        BoundType.BERNSTEIN,
        BoundType.ADAPTIVE
    ]
    
    all_decisions = {}
    
    for bound_type in bound_types:
        print(f"\n📊 Testing {bound_type.value.upper()} bound...")
        
        # Create decision engine with this bound
        engine = DecisionEngine(
            threshold=0.6,
            confidence=0.95,
            default_bound=bound_type,
            verbose=False
        )
        engine.decisions = []  # Clear decisions list
        
        # Test each query
        for i, query in enumerate(queries):
            # Retrieve documents
            results = retriever.retrieve_with_scores(query, k=15)
            scores = [score for _, score in results]
            
            # Make decision
            decision = engine.decide(
                scores=scores,
                query=query,
                generator=generator
            )
            
            # Print progress every 5 queries
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(queries)} queries")
        
        # Store decisions
        all_decisions[bound_type.value] = engine.decisions.copy()
    
    # 4. Calculate metrics
    print("\n📈 Calculating metrics...")
    results = {}
    for name, decisions in all_decisions.items():
        results[name] = MetricsCalculator.from_decisions(decisions)
    
    # 5. Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Answer rate:     {result.answer_rate*100:.1f}%")
        print(f"  Refusal rate:    {result.refusal_rate*100:.1f}%")
        print(f"  Avg lower bound: {result.avg_lower_bound:.3f}")
        print(f"  Avg docs:        {result.avg_docs_retrieved:.1f}")
        print(f"  Bound usage:     {result.bound_counts}")
    
    # 6. Save results
    print("\n💾 Saving results...")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Convert decisions to serializable format
    serializable = {}
    for name, decisions in all_decisions.items():
        serializable[name] = []
        for d in decisions:
            # Convert decision to dict with proper types
            decision_dict = {
                'can_answer': bool(d.can_answer),
                'lower_bound': float(d.lower_bound),
                'bound_type': str(d.bound_type),
                'n_docs': int(d.n_docs),
                'message': str(d.message),
                'metadata': {
                    'mean': float(d.metadata.get('mean', 0)) if d.metadata else 0,
                    'std': float(d.metadata.get('std', 0)) if d.metadata else 0,
                    'scores': [float(s) for s in d.metadata.get('scores', [])] if d.metadata and d.metadata.get('scores') else []
                } if d.metadata else None
            }
            serializable[name].append(decision_dict)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/bound_comparison_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(serializable, f, indent=2)
    
    print(f"✅ Results saved to {filename}")
    
    return results, all_decisions


if __name__ == "__main__":
    results, decisions = run_comparison()
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)