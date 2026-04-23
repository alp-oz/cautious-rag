# Cautious RAG: Probabilistic Hallucination Control

A retrieval-augmented generation pipeline that uses concentration inequalities to abstain from answering when retrieved evidence is statistically insufficient.

## Motivation

Standard RAG systems answer every query, even when retrieved documents provide weak or irrelevant evidence. This produces confident-sounding hallucinations. Cautious RAG addresses this by framing the decision to answer as a hypothesis test: given a sample of retrieval similarity scores, can we statistically certify that the lower bound on true relevance clears a threshold? If not, the system declines.

## Method

For a query with `n` retrieved documents, let `μ̂` denote the sample mean similarity score. We compute a lower confidence bound

```
lower_bound = μ̂ - ε
```

and answer only when `lower_bound ≥ τ` for a fixed threshold `τ`. The slack `ε` is derived from one of three concentration inequalities depending on the retrieval setting:

| Inequality | Assumption | Guarantee |
|---|---|---|
| Hoeffding (1963) | Independent documents | P(\|μ̂ − μ\| ≥ ε) ≤ 2·exp(−2nε²) |
| Azuma (1967) | Sequential / dependent retrieval | P(\|Mₙ\| ≥ ε) ≤ 2·exp(−ε²/2nc²) |
| Bernstein (1924) | Low-variance scores | Variance-aware tighter bound |

The Azuma bound is the natural choice when retrieval is done sequentially or with reranking, as the similarity scores form a martingale difference sequence.

## Usage

```bash
git clone https://github.com/alp-oz/cautious-rag
cd cautious-rag
pip install -e .
```

```python
from cautious_rag import CautiousRAG

rag = CautiousRAG(documents)
result = rag.answer("Who was David Bohm?")

if result.confident:
    print(f"Answer: {result.answer}")
    print(f"Relevance lower bound: {result.lower_bound:.2f} (95% confidence)")
else:
    print(f"Insufficient evidence (lower bound {result.lower_bound:.2f} < threshold {result.threshold:.2f})")
```

## Project Structure

```
cautious-rag/
├── cautious_rag/
│   ├── bounds/             # Concentration inequality implementations
│   │   ├── hoeffding.py
│   │   ├── azuma.py
│   │   └── bernstein.py
│   ├── decision/           # Confidence thresholding logic
│   └── core/               # RAG pipeline components
├── experiments/
│   └── 09_openai_hallucination_test.py
└── README.md
```

## Experiments

The main demo runs on TriviaQA with random sampling and requires an OpenAI API key:

```bash
export OPENAI_KEY="your-key"
cd experiments
python 09_openai_hallucination_test.py
```

Results vary across runs due to random document sampling, which reflects the stochastic nature of the retrieval setting.

## Theoretical Background

The concentration inequalities used here are classical tools from probability theory. The Hoeffding and Azuma bounds appear throughout the PAC learning and online learning literature; the connection to RAG confidence is described in the inline documentation. For the martingale-based bound, the relevant reference is:

- Azuma, K. (1967). Weighted sums of certain dependent random variables. *Tôhoku Mathematical Journal*, 19(3), 357–367.
- Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association*, 58(301), 13–30.
- Bernstein, S. (1924). On a modification of Chebyshev's inequality and of the error formula of Laplace.

## License

MIT
