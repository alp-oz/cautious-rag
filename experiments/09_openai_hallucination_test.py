#!/usr/bin/env python
"""
OpenAI Hallucination Test - REAL random data each time.
FIXED: Actually extracts documents properly.
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import time
import random
from openai import OpenAI
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent.parent))

from cautious_rag.core.retriever import Retriever
from cautious_rag.decision.engine import DecisionEngine, BoundType

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


def load_random_real_data():
    """
    Load RANDOM real questions AND their documents from TriviaQA.
    Different every time you run it.
    """
    print("📚 Loading random TriviaQA data...")
    
    # Load the dataset
    dataset = load_dataset("trivia_qa", "rc", split="validation")
    
    # Pick 5 RANDOM questions (different each time)
    indices = random.sample(range(len(dataset)), 5)
    
    test_queries = []
    all_docs = []
    
    print(f"  Randomly selected {len(indices)} questions")
    
    for i, idx in enumerate(indices):
        item = dataset[idx]
        
        # Get question
        question = item['question']
        
        # Get answer
        answer = item['answer']['value']
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        
        test_queries.append({
            'query': question,
            'correct': answer
        })
        
        print(f"  Q{i+1}: {question[:60]}...")
        
        # ===== FIXED: Better document extraction =====
        docs_added = 0
        
        # Method 1: Entity pages (titles and descriptions)
        if 'entity_pages' in item:
            for page in item['entity_pages']:
                if isinstance(page, dict):
                    # Add title
                    title = page.get('title', '')
                    if title and len(title) > 3:
                        all_docs.append(title)
                        docs_added += 1
                    
                    # Add Wikipedia extract if available
                    if 'wiki' in page and page['wiki']:
                        all_docs.append(page['wiki'])
                        docs_added += 1
        
        # Method 2: Search results
        if 'search_results' in item:
            for result in item['search_results']:
                if isinstance(result, dict):
                    # Add title
                    title = result.get('title', '')
                    if title and len(title) > 3:
                        all_docs.append(title)
                        docs_added += 1
                    
                    # Add description
                    desc = result.get('description', '')
                    if desc and len(desc) > 10:
                        all_docs.append(desc)
                        docs_added += 1
                    
                    # Add search result text if available
                    if 'result' in result and result['result']:
                        all_docs.append(result['result'])
                        docs_added += 1
        
        print(f"     Added {docs_added} documents for this question")
    
    # Remove duplicates and limit
    documents = list(set(all_docs))
    
    # Filter out very short documents
    documents = [doc for doc in documents if len(doc) > 20]
    
    print(f"\n✅ Total unique documents: {len(documents)}")
    
    # If we still have no documents, use fallback
    if len(documents) < 10:
        print("⚠️  Not enough documents from TriviaQA, using fallback...")
        documents = [
            "Margaret Thatcher was Prime Minister from 1979 to 1990. John Major was Chancellor of the Exchequer from 1989 to 1990, then became Prime Minister.",
            "Geoffrey Howe was Chancellor from 1979 to 1983. Nigel Lawson was Chancellor from 1983 to 1989. John Major was Chancellor from 1989 to 1990.",
            "Chamomile tea is known for its calming properties and is often used to aid relaxation and sleep.",
            "Peppermint tea can help with digestion and relaxation. Lavender tea is also used for stress relief.",
            "Room 101 was a BBC comedy series that started in 1994. The first presenter was Nick Hancock.",
            "The Ffestiniog Railway is a narrow-gauge railway in Wales. The western end station is Porthmadog Harbour.",
            "Porthmadog is a town in Gwynedd, Wales. It is the western terminus of the Ffestiniog Railway.",
            "Association football rules were codified in England in 1863. The birthplace is often considered to be the Freemasons' Tavern in London.",
            "The Football Association was founded in 1863 in London. The Freemasons' Tavern on Great Queen Street hosted the meetings.",
        ]
        print(f"✅ Using {len(documents)} fallback documents")
    
    print(f"✅ Final document count: {len(documents)}")
    
    return documents, test_queries


def check_answer(answer, correct):
    """Check if answer contains correct info."""
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


def generate_answer(query, docs):
    """Generate answer using OpenAI."""
    if not docs:
        return "No documents available"
    
    context = "\n".join(docs[:5])  # Use top 5 docs
    
    prompt = f"""Based ONLY on these documents, answer the question.

Documents:
{context}

Question: {query}

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer based only on the documents. Be concise."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def run_test(threshold=0.2):
    """Run test with random real data."""
    print("=" * 60)
    print("OPENAI TEST - RANDOM REAL DATA")
    print("=" * 60)
    
    # Load random data
    documents, test_queries = load_random_real_data()
    
    if len(documents) < 5:
        print("❌ Not enough documents to run test")
        return
    
    # Initialize
    retriever = Retriever(documents)
    engine = DecisionEngine(
        threshold=threshold,
        confidence=0.95,
        default_bound=BoundType.ADAPTIVE
    )
    
    results = {
        'queries': [],
        'standard_correct': [],
        'cautious_correct': [],
        'cautious_refused': [],
        'lower_bounds': []
    }
    
    print(f"\n📋 Testing {len(test_queries)} random queries...\n")
    
    for i, test in enumerate(test_queries):
        query = test['query']
        correct = test['correct']
        
        print(f"\n--- Query {i+1} ---")
        print(f"Q: {query}")
        print(f"Correct: {correct}")
        
        # Retrieve
        docs_with_scores = retriever.retrieve_with_scores(query, k=10)
        docs = [doc for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        
        # Show top docs
        print(f"\nTop docs:")
        for j, (doc, score) in enumerate(zip(docs[:2], scores[:2])):
            print(f"  {j+1}. [{score:.3f}] {doc[:80]}...")
        
        # Standard answer
        print("\n  🤖 Generating standard answer...")
        standard_answer = generate_answer(query, docs)
        standard_correct = check_answer(standard_answer, correct)
        print(f"  Standard: {'✅' if standard_correct else '❌'}")
        print(f"    {standard_answer[:100]}...")
        
        # Cautious decision
        decision = engine.decide(scores)
        lb = decision.lower_bound
        
        if decision.can_answer:
            print(f"\n  🤖 Generating cautious answer (LB={lb:.3f})...")
            cautious_answer = generate_answer(query, docs)
            cautious_correct = check_answer(cautious_answer, correct)
            print(f"  Cautious: {'✅' if cautious_correct else '❌'}")
            print(f"    {cautious_answer[:100]}...")
        else:
            cautious_correct = False
            print(f"\n  Cautious: 🤔 REFUSED (LB={lb:.3f})")
        
        # Store results
        results['queries'].append(query)
        results['standard_correct'].append(standard_correct)
        results['cautious_correct'].append(cautious_correct)
        results['cautious_refused'].append(not decision.can_answer)
        results['lower_bounds'].append(lb)
        
        time.sleep(1)  # Rate limiting
    
    # Analyze
    total = len(test_queries)
    standard_hallucinations = sum(1 for x in results['standard_correct'] if not x)
    standard_rate = standard_hallucinations / total
    
    cautious_answered = sum(1 for i in range(total) if not results['cautious_refused'][i])
    cautious_hallucinations = sum(1 for i in range(total) 
                                 if not results['cautious_refused'][i] 
                                 and not results['cautious_correct'][i])
    
    if cautious_answered > 0:
        cautious_rate = cautious_hallucinations / cautious_answered
    else:
        cautious_rate = 0
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Standard RAG:")
    print(f"  Correct: {total - standard_hallucinations}/{total}")
    print(f"  Hallucinations: {standard_hallucinations}/{total} ({standard_rate:.1%})")
    
    print(f"\n🛡️ Cautious RAG (threshold={threshold}):")
    print(f"  Answered: {cautious_answered}/{total}")
    print(f"  Refused: {total - cautious_answered}/{total}")
    if cautious_answered > 0:
        print(f"  Hallucinations when answered: {cautious_hallucinations}/{cautious_answered} ({cautious_rate:.1%})")
    
    if standard_rate > 0 and cautious_answered > 0:
        reduction = (standard_rate - cautious_rate) / standard_rate * 100
        print(f"\n🎯 Hallucination reduction: {reduction:.1f}%")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar chart
    x = range(total)
    width = 0.35
    ax1.bar([i - width/2 for i in x], results['standard_correct'], width, label='Standard', color='blue', alpha=0.7)
    ax1.bar([i + width/2 for i in x], results['cautious_correct'], width, label='Cautious', color='green', alpha=0.7)
    ax1.set_xlabel('Query')
    ax1.set_ylabel('Correct (1) / Incorrect (0)')
    ax1.set_title('Standard vs Cautious')
    ax1.legend()
    ax1.set_ylim([0, 1.2])
    
    # Lower bounds
    colors = ['green' if not r else 'red' for r in results['cautious_refused']]
    ax2.bar(x, results['lower_bounds'], color=colors, alpha=0.7)
    ax2.axhline(y=threshold, color='blue', linestyle='--', label=f'Threshold ({threshold})')
    ax2.set_xlabel('Query')
    ax2.set_ylabel('Lower Bound')
    ax2.set_title('Cautious RAG Decisions')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/openai_random_test.png')
    plt.show()
    
    return results


if __name__ == "__main__":
    print("🚀 OpenAI Test with RANDOM real data")
    print("⚠️  This costs money (~$0.02 per run)")
    
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        results = run_test(threshold=0.2)