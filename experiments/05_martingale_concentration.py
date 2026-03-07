#!/usr/bin/env python
"""
Experiment 5: Martingale Concentration for RAG
Compare martingale-based bounds with independence assumption.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from cautious_rag.bounds.martingale_concentration import (
    MartingaleConcentration, 
    AdaptiveMartingaleConcentration
)


# ============================================
# CONFIGURATION - CHANGE THESE PARAMETERS
# ============================================

CONFIG = {
    # Experiment parameters
    'n_trials': 200,        # Number of sequences to test
    'n_docs': 30,           # Number of documents per sequence
    
    # Bound parameters
    'confidence': 0.95,      # Confidence level (try 0.90, 0.85, 0.80)
    'fixed_max_change': 0.05, # Max change for fixed Azuma (try 0.10, 0.08, 0.05)
    
    # Data generation
    'structures': {
        'strong_decay': {
            'name': 'Strong Dependence',
            'base': 0.95, 
            'decay': 0.85, 
            'noise': 0.03
        },
        'moderate_decay': {
            'name': 'Moderate Dependence',
            'base': 0.9, 
            'decay': 0.9,   
            'noise': 0.05
        },
        'weak_decay': {
            'name': 'Weak Dependence',
            'base': 0.85, 
            'decay': 0.95,  
            'noise': 0.08
        },
        'independent': {
            'name': 'Nearly Independent',
            'base': 0.8, 
            'decay': 0.98,  
            'noise': 0.1
        }
    }
}
# ============================================


def generate_dependent_sequence(n_docs: int = 30, 
                               base: float = 0.9, 
                               decay: float = 0.85, 
                               noise: float = 0.05,
                               seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a dependent sequence mimicking RAG retrieval.
    """
    if seed is not None:
        np.random.seed(seed)
    
    scores = []
    for i in range(n_docs):
        if i == 0:
            score = base + noise * np.random.randn()
        else:
            score = scores[-1] * decay + noise * np.random.randn()
        scores.append(np.clip(score, 0, 1))
    
    return np.array(scores)


def run_martingale_experiment():
    """
    Run experiment comparing different concentration bounds.
    Uses parameters from CONFIG at the top of the file.
    """
    print("=" * 60)
    print("EXPERIMENT 5: Martingale Concentration for RAG")
    print("=" * 60)
    print(f"\n📋 CONFIGURATION:")
    print(f"   n_trials = {CONFIG['n_trials']}")
    print(f"   n_docs = {CONFIG['n_docs']}")
    print(f"   confidence = {CONFIG['confidence']}")
    print(f"   fixed_max_change = {CONFIG['fixed_max_change']}")
    print("=" * 60)
    
    structures = CONFIG['structures']
    all_results = {}
    
    for struct_key, struct_params in structures.items():
        print(f"\n📊 Testing: {struct_params['name']}")
        
        # Track results
        bounds = {
            'hoeffding': [],
            'azuma_fixed': [],
            'azuma_adaptive': []
        }
        coverages = {
            'hoeffding': 0,
            'azuma_fixed': 0,
            'azuma_adaptive': 0
        }
        true_means = []
        
        # Initialize bound calculators with confidence from CONFIG
        mc = MartingaleConcentration(confidence=CONFIG['confidence'])
        adaptive_mc = AdaptiveMartingaleConcentration(confidence=CONFIG['confidence'])
        
        for trial in range(CONFIG['n_trials']):
            # Generate sequence
            scores = generate_dependent_sequence(
                n_docs=CONFIG['n_docs'],
                base=struct_params['base'],
                decay=struct_params['decay'],
                noise=struct_params['noise'],
                seed=trial if trial < 100 else None
            )
            
            true_mean = np.mean(scores)
            true_means.append(true_mean)
            
            # 1. Hoeffding (assumes independence)
            n = len(scores)
            delta = 1 - CONFIG['confidence']
            hoeffding_eps = np.sqrt(np.log(2 / delta) / (2 * n))
            hoeffding_lower = np.mean(scores) - hoeffding_eps
            
            # 2. Azuma with fixed max_change from CONFIG
            azuma_fixed = mc.azuma_hoeffding_bound(scores, max_change=CONFIG['fixed_max_change'])
            
            # 3. Adaptive Azuma
            azuma_adaptive = adaptive_mc.bound_with_adaptive_c(scores)
            
            # Store bounds
            bounds['hoeffding'].append(hoeffding_lower)
            bounds['azuma_fixed'].append(azuma_fixed['lower_bound'])
            bounds['azuma_adaptive'].append(azuma_adaptive['lower_bound'])
            
            # Check coverage (does lower bound ≤ true mean?)
            if hoeffding_lower <= true_mean:
                coverages['hoeffding'] += 1
            if azuma_fixed['lower_bound'] <= true_mean:
                coverages['azuma_fixed'] += 1
            if azuma_adaptive['lower_bound'] <= true_mean:
                coverages['azuma_adaptive'] += 1
            
            # Print progress
            if (trial + 1) % 50 == 0:
                print(f"  Processed {trial + 1}/{CONFIG['n_trials']} trials")
        
        # Calculate coverage rates
        coverage_rates = {
            k: v / CONFIG['n_trials'] for k, v in coverages.items()
        }
        
        # Average bounds
        avg_bounds = {
            k: np.mean(v) for k, v in bounds.items()
        }
        
        # Store results
        all_results[struct_key] = {
            'name': struct_params['name'],
            'coverage': coverage_rates,
            'avg_bounds': avg_bounds,
            'avg_true_mean': np.mean(true_means)
        }
        
        # Print results for this structure
        print(f"\n  Results for {struct_params['name']}:")
        print(f"  Average true mean: {np.mean(true_means):.3f}")
        print(f"  Coverage (target = {CONFIG['confidence']}):")
        for method, rate in coverage_rates.items():
            status = "✓" if rate >= CONFIG['confidence'] else "❌"
            print(f"    {method:15s}: {rate:.3f} {status}")
        print(f"  Average lower bounds:")
        for method, bound in avg_bounds.items():
            print(f"    {method:15s}: {bound:.3f}")
    
    return all_results


def plot_results(results: Dict, save_path: Optional[str] = None):
    """
    Plot the experiment results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    structures = list(results.keys())
    methods = ['hoeffding', 'azuma_fixed', 'azuma_adaptive']
    colors = {'hoeffding': 'red', 'azuma_fixed': 'blue', 'azuma_adaptive': 'green'}
    
    # Plot 1: Coverage rates
    ax1 = axes[0, 0]
    x = np.arange(len(structures))
    width = 0.25
    
    for i, method in enumerate(methods):
        coverages = [results[s]['coverage'][method] for s in structures]
        ax1.bar(x + i*width - width, coverages, width, 
                label=method, color=colors[method], alpha=0.7)
    
    ax1.axhline(y=CONFIG['confidence'], color='black', linestyle='--', 
                label=f'Target ({CONFIG["confidence"]})')
    ax1.set_xlabel('Dependence Structure')
    ax1.set_ylabel('Coverage Rate')
    ax1.set_title('Coverage: Should be ≥ Target')
    ax1.set_xticks(x)
    ax1.set_xticklabels([results[s]['name'] for s in structures], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average lower bounds
    ax2 = axes[0, 1]
    for method in methods:
        bounds = [results[s]['avg_bounds'][method] for s in structures]
        ax2.plot(structures, bounds, 'o-', label=method, color=colors[method], linewidth=2)
    
    true_means = [results[s]['avg_true_mean'] for s in structures]
    ax2.plot(structures, true_means, 's--', label='true mean', color='purple', linewidth=2)
    
    ax2.set_xlabel('Dependence Structure')
    ax2.set_ylabel('Average Lower Bound')
    ax2.set_title('Tightness: Higher is Better')
    ax2.set_xticks(range(len(structures)))
    ax2.set_xticklabels([results[s]['name'] for s in structures], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coverage deficit
    ax3 = axes[1, 0]
    hoeffding_deficit = [max(0, CONFIG['confidence'] - results[s]['coverage']['hoeffding']) for s in structures]
    azuma_deficit = [max(0, CONFIG['confidence'] - results[s]['coverage']['azuma_fixed']) for s in structures]
    adaptive_deficit = [max(0, CONFIG['confidence'] - results[s]['coverage']['azuma_adaptive']) for s in structures]
    
    x = np.arange(len(structures))
    ax3.bar(x - 0.25, hoeffding_deficit, 0.25, label='Hoeffding deficit', color='red', alpha=0.7)
    ax3.bar(x, azuma_deficit, 0.25, label='Azuma fixed deficit', color='blue', alpha=0.7)
    ax3.bar(x + 0.25, adaptive_deficit, 0.25, label='Adaptive deficit', color='green', alpha=0.7)
    
    ax3.set_xlabel('Dependence Structure')
    ax3.set_ylabel('Coverage Deficit (below target)')
    ax3.set_title('Lower is Better (0 = perfect)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([results[s]['name'] for s in structures], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency
    ax4 = axes[1, 1]
    for method in methods:
        efficiencies = []
        for s in structures:
            if results[s]['coverage'][method] >= CONFIG['confidence'] - 0.02:  # Close enough
                efficiency = results[s]['avg_bounds'][method] / results[s]['avg_true_mean']
                efficiencies.append(efficiency)
            else:
                efficiencies.append(0)
        
        ax4.plot(structures, efficiencies, 'o-', label=method, color=colors[method], linewidth=2)
    
    ax4.set_xlabel('Dependence Structure')
    ax4.set_ylabel('Efficiency (bound / true mean)')
    ax4.set_title('Tightness: Higher is Better (if coverage OK)')
    ax4.set_xticks(range(len(structures)))
    ax4.set_xticklabels([results[s]['name'] for s in structures], rotation=45, ha='right')
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Plot saved to {save_path}")
    
    plt.show()


def main():
    """Run the martingale concentration experiment."""
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run experiment
    results = run_martingale_experiment()
    
    # Plot results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"results/martingale_concentration_{timestamp}.png"
    plot_results(results, save_path=plot_path)
    
    # Save results
    with open(f"results/martingale_concentration_{timestamp}.json", "w") as f:
        serializable = {}
        for k, v in results.items():
            serializable[k] = {
                'name': v['name'],
                'coverage': {k2: float(v2) for k2, v2 in v['coverage'].items()},
                'avg_bounds': {k2: float(v2) for k2, v2 in v['avg_bounds'].items()},
                'avg_true_mean': float(v['avg_true_mean'])
            }
        json.dump(serializable, f, indent=2)
    
    print("\n✅ Experiment complete!")
    print(f"📊 Plot: {plot_path}")
    print(f"📊 Config used: confidence={CONFIG['confidence']}, "
          f"max_change={CONFIG['fixed_max_change']}, n_docs={CONFIG['n_docs']}")


if __name__ == "__main__":
    main()