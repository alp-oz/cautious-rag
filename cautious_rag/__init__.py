"""
Cautious RAG: A retrieval system that knows when not to answer.
"""

__version__ = "0.1.0"  # This line imports from version.py

from .core.retriever import Retriever
from .core.generator import Generator
from .decision.engine import DecisionEngine, BoundType
from .bounds.hoeffding import HoeffdingBound
from .bounds.azuma import AzumaBound
from .bounds.bernstein import EmpiricalBernsteinBound

__all__ = [
    'Retriever',
    'Generator',
    'DecisionEngine',
    'BoundType',
    'HoeffdingBound',
    'AzumaBound',
    'EmpiricalBernsteinBound',
]