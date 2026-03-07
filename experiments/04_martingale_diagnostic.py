#!/usr/bin/env python
"""
Experiment 4: Martingale Diagnostic for RAG Dependence Structure.
Run with: python experiments/04_martingale_diagnostic.py

Measures how the next document's relevance depends on past average."""
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from pathlib import Path
from scipy import stats

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from cautious_rag.utils.data import DocumentCollection
from cautious_rag.core.retriever import Retriever


class MartingaleDiagnostic:
    """
    Diagnose the dependence structure in sequential retrieval.
    
    In permutation descents (your paper):
        E[D_{n+1} | past] = (n/(n+1)) D_n + n/(n+1)
    
    For RAG, we want to find:
        E[X_{n+1} | past] = a_n * R_n + b_n
    where:
        X_{n+1} = relevance of next document
        R_n = average relevance of first n documents
        a_n, b_n = coefficients to estimate
    """
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.data = []  # Store (n, past_avg, next_score, all_scores)
        
    def run_diagnostic(self, queries, k_max=20):
        """
        Run diagnostic on multiple queries.
        
        Args:
            queries: List of query strings
            k_max: Maximum number of documents to retrieve
        """
        print(f"\n🔍 Running martingale diagnostic on {len(queries)} queries...")
        
        for q_idx, query in enumerate(queries):
            # Retrieve documents with scores
            results = self.retriever.retrieve_with_scores(query, k=k_max)
            scores = [score for _, score in results]
            
            # For each step, record relationship
            for i in range(1, len(scores)):
                past_scores = scores[:i]
                past_avg = np.mean(past_scores)
                next_score = scores[i]
                
                self.data.append({
                    'query_idx': q_idx,
                    'n': i,  # number of past documents
                    'past_avg': past_avg,
                    'next_score': next_score,
                    'all_scores': scores.copy()
                })
            
            if (q_idx + 1) % 10 == 0:
                print(f"  Processed {q_idx + 1}/{len(queries)} queries")
        
        print(f"✅ Collected {len(self.data)} data points")
        return self.data
    
    def estimate_coefficients(self):
        """
        Estimate a_n and b_n for each n.
        
        Returns:
            coefficients: Dict with 'a_n', 'b_n', 'r_squared' for each n
        """
        # Group by n
        data_by_n = {}
        for point in self.data:
            n = point['n']
            if n not in data_by_n:
                data_by_n[n] = {'past_avgs': [], 'next_scores': []}
            data_by_n[n]['past_avgs'].append(point['past_avg'])
            data_by_n[n]['next_scores'].append(point['next_score'])
        
        # Print debug info for first few n
        print("\n📊 Data variation by n:")
        for n in sorted(data_by_n.keys())[:5]:
            x = data_by_n[n]['past_avgs']
            print(f"  n={n}: range = {min(x):.3f} to {max(x):.3f}, unique values = {len(set(x))}")
        
        # Estimate coefficients for each n
        coefficients = {}
        for n in sorted(data_by_n.keys()):
            x = np.array(data_by_n[n]['past_avgs'])
            y = np.array(data_by_n[n]['next_scores'])
            
            if len(x) >= 5:  # Need enough data
                # Check if all x values are the same
                if np.all(x == x[0]):
                    # All past averages are identical!
                    # Then the best estimate of next score is just the mean of y
                    coefficients[n] = {
                        'a_n': 0.0,  # No dependence when x is constant
                        'b_n': np.mean(y),
                        'r_squared': 0.0,
                        'p_value': 1.0,
                        'std_err': np.std(y) / np.sqrt(len(y)),
                        'n_samples': len(x),
                        'note': 'All x identical - no dependence detected'
                    }
                else:
                    # Normal case: x values vary
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        
                        coefficients[n] = {
                            'a_n': slope,
                            'b_n': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'std_err': std_err,
                            'n_samples': len(x)
                        }
                    except Exception as e:
                        # If linregress fails for any other reason
                        coefficients[n] = {
                            'a_n': 0.0,
                            'b_n': np.mean(y),
                            'r_squared': 0.0,
                            'p_value': 1.0,
                            'std_err': np.std(y) / np.sqrt(len(y)),
                            'n_samples': len(x),
                            'note': f'Linregress failed: {str(e)}'
                        }
        
        return coefficients
    
    def compare_to_permutation(self):
        """
        Compare RAG dependence to permutation descents.
        
        In permutations: a_n = n/(n+1) ≈ 1 - 1/(n+1)
        """
        permutation_a_n = {n: n/(n+1) for n in range(1, 20)}
        return permutation_a_n
    
    def plot_dependence(self, coefficients, save_path=None):
        """Plot the estimated dependence structure."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Scatter plot of past_avg vs next_score
        ax1 = axes[0, 0]
        n_values = [point['n'] for point in self.data]
        past_avgs = [point['past_avg'] for point in self.data]
        next_scores = [point['next_score'] for point in self.data]
        
        scatter = ax1.scatter(past_avgs, next_scores, c=n_values, cmap='viridis', 
                              alpha=0.5, s=10)
        ax1.set_xlabel('Average relevance of past documents (R_n)')
        ax1.set_ylabel('Relevance of next document (X_{n+1})')
        ax1.set_title('Dependence in Sequential Retrieval')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='n (number of past docs)')
        
        # Add diagonal line y=x for reference
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x (no dependence)')
        ax1.legend()
        
        # Plot 2: a_n vs n (RAG vs Permutation)
        ax2 = axes[0, 1]
        n_vals = sorted([n for n in coefficients.keys() if n <= 15])
        a_vals = [coefficients[n]['a_n'] for n in n_vals]
        a_errors = [coefficients[n].get('std_err', 0) for n in n_vals]
        
        # RAG a_n
        ax2.errorbar(n_vals, a_vals, yerr=a_errors, fmt='o-', 
                     label='RAG (estimated)', capsize=5)
        
        # Permutation a_n = n/(n+1)
        perm_n = np.arange(1, 16)
        perm_a = perm_n / (perm_n + 1)
        ax2.plot(perm_n, perm_a, 'r--', label='Permutation: n/(n+1)')
        
        ax2.set_xlabel('n (number of past documents)')
        ax2.set_ylabel('a_n (dependence coefficient)')
        ax2.set_title('Dependence Strength: RAG vs Permutations')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: b_n vs n
        ax3 = axes[1, 0]
        b_vals = [coefficients[n]['b_n'] for n in n_vals]
        
        ax3.plot(n_vals, b_vals, 'o-', color='green')
        ax3.set_xlabel('n')
        ax3.set_ylabel('b_n (baseline coefficient)')
        ax3.set_title('Baseline Coefficient b_n')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: R-squared vs n (goodness of fit)
        ax4 = axes[1, 1]
        r2_vals = [coefficients[n]['r_squared'] for n in n_vals]
        
        ax4.bar(n_vals, r2_vals, alpha=0.7)
        ax4.set_xlabel('n')
        ax4.set_ylabel('R² (goodness of fit)')
        ax4.set_title('How well does linear model fit?')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self, coefficients):
        """Print a summary of findings."""
        print("\n" + "=" * 60)
        print("MARTINGALE DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        # Filter out coefficients with note (like constant x)
        valid_coeffs = {n: c for n, c in coefficients.items() if 'note' not in c}
        
        if not valid_coeffs:
            print("\n⚠️  No valid coefficients estimated yet.")
            print("   Try running with more queries to get variation in past averages.")
            return
        
        # Average dependence
        avg_a = np.mean([c['a_n'] for c in valid_coeffs.values()])
        avg_b = np.mean([c['b_n'] for c in valid_coeffs.values()])
        
        print(f"\n📈 Average dependence coefficient a_n: {avg_a:.3f}")
        print(f"   (If a_n ≈ 0: independent, a_n > 0: positive dependence, a_n < 0: negative dependence)")
        
        print(f"\n📊 For comparison, permutations have a_n = n/(n+1) ≈ 1 - 1/(n+1)")
        
        print("\nDetailed coefficients by n:")
        print(" n |    a_n    |    b_n    |  R²  | samples")
        print("---|-----------|-----------|------|--------")
        for n in sorted(valid_coeffs.keys())[:10]:  # First 10
            c = valid_coeffs[n]
            print(f"{n:2d} |  {c['a_n']:6.3f}   |  {c['b_n']:6.3f}   | {c['r_squared']:.3f} |  {c['n_samples']:3d}")
        
        # Interpretation
        print("\n🔍 INTERPRETATION:")
        if avg_a > 0.1:
            print("  • Strong positive dependence: Better past → better next")
            print("  • Different from permutations (which have negative dependence)")
            print("  • Would need to adapt your martingale construction")
        elif avg_a < -0.1:
            print("  • Strong negative dependence: Better past → worse next")
            print("  • This matches the permutation case!")
            print("  • Your martingale methods are directly applicable!")
        else:
            print("  • Weak dependence (near independent)")
            print("  • Hoeffding bounds might be sufficient")
    
    def save_results(self, coefficients, filename=None):
        """Save results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/martingale_diagnostic_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        
        # Convert to serializable
        serializable = {
            'coefficients': {
                str(n): {
                    'a_n': float(c.get('a_n', 0)),
                    'b_n': float(c.get('b_n', 0)),
                    'r_squared': float(c.get('r_squared', 0)),
                    'p_value': float(c.get('p_value', 1.0)),
                    'std_err': float(c.get('std_err', 0)),
                    'n_samples': int(c.get('n_samples', 0)),
                    'note': c.get('note', '')
                }
                for n, c in coefficients.items()
            },
            'n_data_points': len(self.data),
            'n_queries': len(set(point['query_idx'] for point in self.data))
        }
        
        # Add summary if there are valid coefficients
        valid_coeffs = {n: c for n, c in coefficients.items() if 'note' not in c}
        if valid_coeffs:
            serializable['summary'] = {
                'avg_a_n': float(np.mean([c['a_n'] for c in valid_coeffs.values()])),
                'avg_b_n': float(np.mean([c['b_n'] for c in valid_coeffs.values()])),
                'avg_r_squared': float(np.mean([c['r_squared'] for c in valid_coeffs.values()]))
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
        return filename


def main():
    """Run the martingale diagnostic."""
    print("=" * 60)
    print("EXPERIMENT 4: Martingale Diagnostic for RAG")
    print("=" * 60)
    print("Directly applies techniques from:")
    print("  Ozturk, A. (2022). Martingales and descent statistics.")
    print("  Journal of Combinatorial Theory, Series A.")
    print("=" * 60)
    
    # 1. Create test data
    print("\n📚 Creating test documents...")
    docs = DocumentCollection()  # Random documents
    queries = docs.get_sample_queries(50)  # Use 50 queries for variation
    
    # 2. Initialize retriever
    print("🔧 Initializing retriever...")
    retriever = Retriever(docs.documents)
    
    # 3. Run diagnostic
    diagnostic = MartingaleDiagnostic(retriever)
    diagnostic.run_diagnostic(queries, k_max=15)
    
    # 4. Estimate coefficients
    coefficients = diagnostic.estimate_coefficients()
    
    # 5. Print summary
    diagnostic.print_summary(coefficients)
    
    # 6. Plot results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"results/martingale_diagnostic_{timestamp}.png"
    diagnostic.plot_dependence(coefficients, save_path=plot_path)
    
    # 7. Save data
    diagnostic.save_results(coefficients)
    
    print("\n✅ Diagnostic complete!")


if __name__ == "__main__":
    main()