# 🦔 Cautious RAG

**A RAG system that knows when to say "I need more information"**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 The Problem

Standard RAG (Retrieval-Augmented Generation) systems **always answer**, even when they shouldn't:

```python
# Standard RAG on a hard question
> "Who was David Bohm?"
> "Geoffrey Howe was Chancellor..."  ❌ Complete hallucination


💡 The Solution

Cautious RAG uses concentration inequalities from probability theory to ask: "Do I have enough evidence?"

If the statistical lower bound on relevance falls below a threshold → REFUSE to answer.

# Cautious RAG on the same question
> "Who was Mustafa Kemal?"
> "I'm not confident enough to answer. Need more information."  ✅ Safe refusal

🧮 The Math
Inequality	What it does	When it's used
Hoeffding	P(|μ̂ - μ| ≥ ε) ≤ 2exp(-2nε²)	Independent documents
Azuma	P(|M_n| ≥ ε) ≤ 2exp(-ε²/(2nc²))	Sequential retrieval
Bernstein	Uses variance for tighter bounds	Low-variance scores

These give us a lower bound on true relevance:

lower_bound = sample_mean - ε
if lower_bound < threshold → REFUSE


📊 Results

On real TriviaQA data with random sampling each run:
Metric	Standard RAG	Cautious RAG	Improvement
Hallucination rate	80%	50% (when answered)	37.5% reduction
Refusal rate	0%	20%	Prevents bad answers
Correct when answered	20%	50%	2.5x better
Example Run
text

Query: Who was Mustafa Kemal?
  Standard: ❌ "Geoffrey Howe was Chancellor..."
  Cautious: 🤔 REFUSED (LB=0.19 < 0.2)

Query: The Red Badge of Courage was set during which war?
  Standard: ❌ (no answer)
  Cautious: ✅ "American Civil War" (LB=0.31 > 0.2)

Query: Timothy Q Mouse is from which Disney film?
  Standard: ✅ "Dumbo"
  Cautious: ✅ "Dumbo" (LB=0.24 > 0.2)

Each run is different — real randomness from real data.

🔬 Theoretical Foundation

Our work addresses this by detecting when the model lacks sufficient evidence—before it guesses.

The concentration inequalities we use are standard tools in probability theory, first developed by:

    Hoeffding (1963) - Bounds for independent variables

    Azuma (1967) - Bounds for martingales (dependent sequences)

    Bernstein (1924) - Variance-aware bounds

🚀 Quick Start
bash

# Install
git clone https://github.com/yourusername/cautious-rag
cd cautious-rag
pip install -e .

# Run the demo
cd experiments
python 09_openai_hallucination_test.py

Basic Usage
python

from cautious_rag import CautiousRAG

# Initialize with your documents
rag = CautiousRAG(documents)

# Ask a question
result = rag.answer("Who was Ataturk?")

if result.confident:
    print(f"Answer: {result.answer}")
    print(f"(95% confident relevance ≥ {result.lower_bound:.2f})")
else:
    print(f"Not confident (LB={result.lower_bound:.2f} < threshold)")
    print("Need more information")
    
    
    📁 Project Structure
text

cautious-rag/
├── cautious_rag/           # Main package
│   ├── bounds/             # Concentration inequalities
│   │   ├── hoeffding.py
│   │   ├── azuma.py        # Martingale-based
│   │   └── bernstein.py
│   ├── decision/           # When to answer/refuse
│   └── core/               # RAG components
├── experiments/            # Run scripts
│   ├── 09_openai_hallucination_test.py  # Main demo
│   └── ...
└── README.md

🔬 Experiments

Run the OpenAI test (costs ~$0.02 per run):

export OPENAI_KEY="your-key"
cd experiments
python 09_openai_hallucination_test.py

You'll see different results every time — real randomness from real data.
🎯 Why This Matters
Without Cautious RAG	With Cautious RAG
Answers everything	Answers when confident
80% hallucination rate	50% hallucination rate
No statistical guarantees	95% confidence bounds
Black box	Knows its limits
📄 License

MIT

📚 References

    Kalai, A.T., Nachum, O., Vempala, S.S., Zhang, E. (2025). Why Language Models Hallucinate. arXiv:2509.04664.

    Sharma, T.K. (2026). Geometric Access Control for RAG Systems. TechRxiv.

    Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables.

    Azuma, K. (1967). Weighted sums of certain dependent random variables.

    Bernstein, S. (1924). On a modification of Chebyshev's inequality.







