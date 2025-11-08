"""
Matching strategy components.
"""

from .exact_matcher import ExactMatcher
from .block_resolver import BlockMappingResolver
from .fuzzy_matcher import FuzzyMatcher

__all__ = ['ExactMatcher', 'BlockMappingResolver', 'FuzzyMatcher']