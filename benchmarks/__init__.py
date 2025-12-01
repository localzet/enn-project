"""
Бенчмарки для сравнения ENN с базовыми моделями
"""

from .compare_models import compare_with_baselines
from .robustness_tests import RobustnessTester

__all__ = [
    'compare_with_baselines',
    'RobustnessTester'
]

