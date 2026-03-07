
**`tests/test_bounds.py`**
```python
"""Test concentration bounds."""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cautious_rag.bounds.hoeffding import HoeffdingBound

def test_hoeffding_basic():
    """Test Hoeffding bound on simple data."""
    bound = HoeffdingBound(confidence=0.95)
    scores = [0.8, 0.7, 0.9, 0.6, 0.85]
    
    lower = bound.lower_bound(scores)
    
    assert 0.0 <= lower <= 1.0
    assert lower <= np.mean(scores)
    
    print("✓ Hoeffding bound test passed")

def test_sample_size():
    """Test sample size calculation."""
    bound = HoeffdingBound(confidence=0.95)
    
    n = bound.sample_size_needed(gap=0.1)
    assert n > 0
    
    # For gap=0.1, should need ~185 samples
    print(f"  For gap=0.1, need {n} samples")
    print("✓ Sample size test passed")

if __name__ == "__main__":
    test_hoeffding_basic()
    test_sample_size()