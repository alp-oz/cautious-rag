#!/usr/bin/env python
"""
Interesting TriviaQA test - shows actual tradeoffs.
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from cautious_rag.core.retriever import Retriever
from cautious_rag.decision.engine import DecisionEngine, BoundType


def load_interesting_test():
    """
    Load test data designed to show interesting behavior.
    """
    print("📚 Creating interesting test data...")
    
    # Documents with contradictions and misleading info
    documents = [
        # Correct info
        "Paris is the capital of France. The Eiffel Tower is in Paris.",
        "Rome is the capital of Italy. The Colosseum is in Rome.",
        "London is the capital of England. Big Ben is in London.",
        "William Shakespeare wrote Hamlet and Romeo and Juliet.",
        "Mars is the Red Planet. It has two moons.",
        
        # Misleading info (to cause hallucinations)
        "Florence is the capital of Italy.",  # WRONG!
        "Milan is the capital of Italy.",     # WRONG!
        "Venice is the capital of Italy.",    # WRONG!
        "Charles Dickens wrote Hamlet.",      # WRONG!
        "Jupiter is the Red Planet.",         # WRONG!
        
        # Partial info (low relevance)
        "Italy has a famous colosseum.",
        "France has the Eiffel Tower.",
        "Shakespeare was a playwright.",
        "Mars appears red in the sky.",
    ]
    
    # Queries with different characteristics
    test_queries = [
        # Clear queries (should answer)
        {"query": "What is the capital of France?", "correct": "Paris", "type": "clear"},
        {"query": "What is the capital of Italy?", "correct": "Rome", "type": "clear"},
        {"query": "Who wrote Hamlet?", "correct": "Shakespeare", "type": "clear"},
        {"query": "Which planet is the Red Planet?", "correct": "Mars", "type": "clear"},
        
        # Ambiguous queries (might refuse)
        {"query": "Where is the Colosseum?", "correct": "Rome", "type": "ambiguous"},
        {"query": "What's the capital of Italy?", "correct": "Rome", "type": "ambiguous"},
        
        # Hard queries (should refuse)
        {"query": "Who painted the Mona Lisa?", "correct": "da Vinci", "type": "hard"},
        {"query": "What's the speed of light?", "correct": "299,792", "type": "hard"},
    ]
    
    return documents, test_queries


def check_answer(answer, correct):
    """Simple check if answer contains correct info."""
    if not answer or not correct:
        return False
    
    answer_lower = answer.lower()
    correct_lower = correct.lower()
    
    if correct_lower in answer_lower:
        return True
    
    # Check for key words
    for word in correct_lower.split():
        if len(word) > 3 and word in answer_lower:
            return True
    
    return False


def run_interesting_test(threshold=0.1):
    """Run test that actually shows interesting behavior."""
    print("=" * 60)
    print("INTERESTING TEST - Shows Real Tradeoffs")
    print("=" * 60)
    
    documents, test_queries = load_interesting_test()
    
    print(f"📚 {len(documents)} documents, {len(test_queries)} queries")
    print(f"🎯 Threshold = {threshold}\n")
    
    # Initialize
    retriever = Retriever(documents)
    engine = DecisionEngine(
        threshold=threshold,
        confidence=0.95,
        default_bound=BoundType.ADAPTIVE
    )
    
    results = {
        'standard': {'correct': 0, 'hallucinations': 0},
        'cautious': {'correct': 0, 'hallucinations': 0, 'refused': 0, 'bounds': []}
    }
    
    for i, test in enumerate(test_queries):
        query = test['query']
        correct = test['correct']
        qtype = test['type']
        
        print(f"\n[{qtype.upper()}] Query {i+1}: {query}")
        
        # Retrieve
        docs_with_scores = retriever.retrieve_with_scores(query, k=8)
        docs = [doc for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        
        # Show top documents
        print(f"  Top docs: {[d[:30]+'...' for d in docs[:2]]}")
        print(f"  Top scores: {[f'{s:.2f}' for s in scores[:2]]}")
        
        # Standard RAG
        standard_answer = docs[0]
        standard_correct = check_answer(standard_answer, correct)
        
        if standard_correct:
            results['standard']['correct'] += 1
            print(f"  Standard: ✅")
        else:
            results['standard']['hallucinations'] += 1
            print(f"  Standard: ❌ (said: {standard_answer[:50]}...)")
        
        # Cautious RAG
        decision = engine.decide(scores)
        lb = decision.lower_bound
        results['cautious']['bounds'].append(lb)
        
        if decision.can_answer:
            cautious_answer = docs[0]
            cautious_correct = check_answer(cautious_answer, correct)
            
            if cautious_correct:
                results['cautious']['correct'] += 1
                print(f"  Cautious: ✅ (LB={lb:.3f})")
            else:
                results['cautious']['hallucinations'] += 1
                print(f"  Cautious: ❌ (LB={lb:.3f})")
        else:
            results['cautious']['refused'] += 1
            print(f"  Cautious: 🤔 REFUSED (LB={lb:.3f})")
    
    # Results
    total = len(test_queries)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    std_rate = results['standard']['hallucinations'] / total
    print(f"\n📊 Standard RAG:")
    print(f"  Correct: {results['standard']['correct']}/{total}")
    print(f"  Hallucinations: {results['standard']['hallucinations']}/{total} ({std_rate:.1%})")
    
    cautious_answered = results['cautious']['correct'] + results['cautious']['hallucinations']
    cautious_refused = results['cautious']['refused']
    
    if cautious_answered > 0:
        cautious_rate = results['cautious']['hallucinations'] / cautious_answered
        print(f"\n🛡️ Cautious RAG (threshold={threshold}):")
        print(f"  Answered: {cautious_answered}/{total}")
        print(f"  Refused: {cautious_refused}/{total}")
        print(f"  Correct when answered: {results['cautious']['correct']}/{cautious_answered} ({results['cautious']['correct']/cautious_answered:.1%})")
        print(f"  Hallucination rate: {cautious_rate:.1%}")
        
        if std_rate > 0:
            reduction = (std_rate - cautious_rate) / std_rate * 100
            print(f"\n🎯 Hallucination reduction: {reduction:.1f}%")
    
    # Plot bounds
    plt.figure(figsize=(10, 4))
    bounds = results['cautious']['bounds']
    colors = ['green' if b > threshold else 'red' for b in bounds]
    plt.bar(range(len(bounds)), bounds, color=colors, alpha=0.7)
    plt.axhline(y=threshold, color='blue', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Query')
    plt.ylabel('Lower Bound')
    plt.title('Cautious RAG: Green = Answer, Red = Refuse')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/interesting_test.png')
    plt.show()
    
    return results


if __name__ == "__main__":
    print("🚀 Running interesting test with threshold=0.1...\n")
    results = run_interesting_test(threshold=0.1)