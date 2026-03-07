#!/usr/bin/env python
"""
Measure how much Cautious RAG reduces hallucinations.
Simple, fast, reliable test.
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from cautious_rag.core.retriever import Retriever
from cautious_rag.decision.engine import DecisionEngine, BoundType


def load_test_data():
    """
    Load simple test data - guaranteed to work, runs instantly.
    """
    print("📚 Creating test documents...")
    
    # Documents with clear facts (some misleading ones to cause hallucinations)
    documents = [
        # Capitals
        "Paris is the capital of France. The Eiffel Tower is in Paris.",
        "London is the capital of England. Big Ben is in London.",
        "Berlin is the capital of Germany. The Brandenburg Gate is in Berlin.",
        "Rome is the capital of Italy. The Colosseum is in Rome.",
        "Madrid is the capital of Spain. The Prado Museum is in Madrid.",
        
        # More facts about same places (creates ambiguity)
        "The Louvre Museum is in Paris, France.",
        "The British Museum is in London, England.",
        "The Uffizi Gallery is in Florence, Italy.",  # Not Rome!
        "The Prado Museum is in Madrid, Spain.",
        "The Eiffel Tower is in Paris, France.",
        "Big Ben is in London, England.",
        "The Colosseum is in Rome, Italy.",
        
        # Some misleading info (to cause hallucinations in standard RAG)
        "Florence is the capital of Italy.",  # WRONG! (Rome is correct)
        "Venice is the capital of Italy.",    # WRONG!
        "Milan is the capital of Italy.",     # WRONG!
        "The Vatican is in Rome, Italy.",
        
        # Other facts
        "William Shakespeare wrote Hamlet and Romeo and Juliet.",
        "Charles Dickens wrote Oliver Twist and A Christmas Carol.",
        "Jane Austen wrote Pride and Prejudice.",
        "Mark Twain wrote The Adventures of Tom Sawyer.",
        
        # More misleading
        "Shakespeare wrote The Odyssey.",  # WRONG! (Homer wrote it)
        "Dickens wrote Les Misérables.",   # WRONG! (Hugo wrote it)
    ]
    
    # Test queries with correct answers
    test_queries = [
        {"query": "What is the capital of Italy?", "correct": "Rome"},
        {"query": "Where is the Colosseum?", "correct": "Rome"},
        {"query": "What is the capital of France?", "correct": "Paris"},
        {"query": "Where is the Louvre Museum?", "correct": "Paris"},
        {"query": "Who wrote Hamlet?", "correct": "Shakespeare"},
        {"query": "What is the capital of Spain?", "correct": "Madrid"},
        {"query": "Where is the Prado Museum?", "correct": "Madrid"},
        {"query": "Who wrote Oliver Twist?", "correct": "Dickens"},
        {"query": "What is the capital of Germany?", "correct": "Berlin"},
        {"query": "Where is the Brandenburg Gate?", "correct": "Berlin"},
    ]
    
    print(f"✅ Created {len(documents)} documents")
    print(f"✅ Created {len(test_queries)} test queries")
    
    return documents, test_queries


def check_answer_simple(answer, correct):
    """
    Simple check if answer contains correct information.
    """
    answer_lower = answer.lower()
    correct_lower = correct.lower()
    
    # Handle common variations
    if correct_lower in answer_lower:
        return True
    
    # Check for partial matches
    correct_words = correct_lower.split()
    for word in correct_words:
        if len(word) > 3 and word in answer_lower:  # Only check significant words
            return True
    
    return False


def run_test(threshold=0.5):
    """
    Run test comparing standard RAG vs Cautious RAG.
    """
    print("=" * 60)
    print("HALLUCINATION TEST")
    print("=" * 60)
    
    # Load test data
    documents, test_queries = load_test_data()
    
    # Initialize retriever
    print("\n🔧 Initializing retriever...")
    retriever = Retriever(documents)
    
    # Results tracking
    results = {
        'standard': {
            'correct': 0,
            'hallucinations': 0,
            'answers': []
        },
        'cautious': {
            'correct': 0,
            'hallucinations': 0,
            'refused': 0,
            'lower_bounds': [],
            'answers': []
        }
    }
    
    # Decision engine
    engine = DecisionEngine(
        threshold=threshold,
        confidence=0.95,
        default_bound=BoundType.ADAPTIVE
    )
    
    print(f"\n📋 Testing {len(test_queries)} queries...\n")
    
    for i, test in enumerate(test_queries):
        query = test['query']
        correct = test['correct']
        
        print(f"Query {i+1}: {query}")
        
        # Retrieve documents
        docs_with_scores = retriever.retrieve_with_scores(query, k=10)
        docs = [doc for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        
        # Standard RAG (always answers with top document)
        standard_answer = docs[0]
        standard_correct = check_answer_simple(standard_answer, correct)
        
        if standard_correct:
            results['standard']['correct'] += 1
            status = '✅'
        else:
            results['standard']['hallucinations'] += 1
            status = '❌'
        
        results['standard']['answers'].append({
            'query': query,
            'answer': standard_answer[:50],
            'correct': standard_correct
        })
        
        print(f"  Standard: {status} {standard_answer[:60]}...")
        
        # Cautious RAG
        decision = engine.decide(scores)
        lower_bound = decision.lower_bound
        results['cautious']['lower_bounds'].append(lower_bound)
        
        if decision.can_answer:
            cautious_answer = docs[0]
            cautious_correct = check_answer_simple(cautious_answer, correct)
            
            if cautious_correct:
                results['cautious']['correct'] += 1
                status = '✅'
            else:
                results['cautious']['hallucinations'] += 1
                status = '❌'
            
            print(f"  Cautious: {status} (LB={lower_bound:.3f}) {cautious_answer[:40]}...")
        else:
            results['cautious']['refused'] += 1
            print(f"  Cautious: 🤔 REFUSED (LB={lower_bound:.3f})")
        
        print()
    
    # Calculate metrics
    total = len(test_queries)
    
    standard_hallucination_rate = results['standard']['hallucinations'] / total
    
    cautious_answered = results['cautious']['correct'] + results['cautious']['hallucinations']
    
    if cautious_answered > 0:
        cautious_hallucination_rate = results['cautious']['hallucinations'] / cautious_answered
    else:
        cautious_hallucination_rate = 0
    
    # Calculate reduction
    if standard_hallucination_rate > 0:
        reduction = (standard_hallucination_rate - cautious_hallucination_rate) / standard_hallucination_rate * 100
    else:
        reduction = 0
    
    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Standard RAG (always answers):")
    print(f"  Correct: {results['standard']['correct']}/{total} ({results['standard']['correct']/total:.1%})")
    print(f"  Hallucinations: {results['standard']['hallucinations']}/{total} ({standard_hallucination_rate:.1%})")
    
    print(f"\n🛡️ Cautious RAG (threshold={threshold}):")
    print(f"  Answered: {cautious_answered}/{total} ({cautious_answered/total:.1%})")
    print(f"  Refused: {results['cautious']['refused']}/{total} ({results['cautious']['refused']/total:.1%})")
    
    if cautious_answered > 0:
        correct_pct = results['cautious']['correct'] / cautious_answered
        print(f"  Correct when answered: {results['cautious']['correct']}/{cautious_answered} ({correct_pct:.1%})")
    else:
        print(f"  Correct when answered: 0/0 (N/A - no answers)")
    
    print(f"  Hallucination rate: {cautious_hallucination_rate:.1%}")
    
    if standard_hallucination_rate > 0:
        print(f"\n🎯 Hallucination reduction: {reduction:.1f}%")
    
    return results, reduction


def plot_results(results, threshold):
    """Create visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    total = len(results['cautious']['lower_bounds'])
    cautious_answered = results['cautious']['correct'] + results['cautious']['hallucinations']
    
    # Plot 1: Answer distribution
    ax1.bar(['Standard RAG'], [results['standard']['correct']], 
            label='Correct', color='green', alpha=0.7)
    ax1.bar(['Standard RAG'], [results['standard']['hallucinations']], 
            bottom=[results['standard']['correct']],
            label='Hallucination', color='red', alpha=0.7)
    
    ax1.bar(['Cautious RAG'], [results['cautious']['correct']], 
            label='Correct', color='green', alpha=0.7)
    ax1.bar(['Cautious RAG'], [results['cautious']['hallucinations']], 
            bottom=[results['cautious']['correct']],
            label='Hallucination', color='red', alpha=0.7)
    ax1.bar(['Cautious RAG'], [results['cautious']['refused']], 
            bottom=[results['cautious']['correct'] + results['cautious']['hallucinations']],
            label='Refused (safe)', color='gray', alpha=0.7)
    
    ax1.set_ylabel('Number of queries')
    ax1.set_title('Response Quality')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hallucination rates
    standard_rate = results['standard']['hallucinations'] / total
    if cautious_answered > 0:
        cautious_rate = results['cautious']['hallucinations'] / cautious_answered
    else:
        cautious_rate = 0
    
    bars = ax2.bar(['Standard RAG', 'Cautious RAG (when answered)'],
                   [standard_rate, cautious_rate],
                   color=['red', 'green'], alpha=0.7)
    
    reduction = (standard_rate - cautious_rate) / standard_rate * 100 if standard_rate > 0 else 0
    ax2.set_ylabel('Hallucination Rate')
    ax2.set_title(f'Cautious RAG: {reduction:.1f}% Fewer Hallucinations')
    ax2.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom')
    
    # Add refusal info
    if results['cautious']['refused'] > 0:
        refusal_rate = results['cautious']['refused'] / total
        ax2.text(0.5, -0.15, f'Cautious RAG refused {results["cautious"]["refused"]}/{total} queries ({refusal_rate:.1%})',
                transform=ax2.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('results/hallucination_test.png', dpi=150)
    plt.show()
    
    print("\n📊 Plot saved to results/hallucination_test.png")


if __name__ == "__main__":
    print("Running hallucination test...")
    # Try with lower threshold so it actually answers some queries
    results, reduction = run_test(threshold=0.3)  # Changed from 0.5 to 0.3
    plot_results(results, threshold=0.3)
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)