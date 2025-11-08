"""
Block mapping resolution for unmatched blocks.

This module provides functionality to resolve block mappings for entities
that don't have exact matches, using block name standardization and fuzzy matching.
"""

import pandas as pd
import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from rapidfuzz import fuzz, process
from tqdm import tqdm

from ..models import EntityRecord, LGDRecord
from ..utils.data_utils import safe_string_conversion, is_null_or_empty


class BlockMappingResolver:
    """
    Resolves block mappings for entities that don't have exact matches.
    
    This class provides fallback matching strategies for blocks using
    name standardization, fuzzy matching, and alternative matching approaches.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, fuzzy_threshold: int = 85):
        """
        Initialize the BlockMappingResolver.
        
        Args:
            logger: Optional logger instance for logging operations
            fuzzy_threshold: Minimum fuzzy matching score for block name matching
        """
        self.logger = logger or logging.getLogger(__name__)
        self.fuzzy_threshold = fuzzy_threshold
        self._block_name_cache = {}
        self._standardization_cache = {}
        self._resolution_stats = {
            'total_processed': 0,
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'unresolved': 0,
            'standardization_applied': 0
        }
    
    def resolve_block_mappings(self, unmatched_entities: pd.DataFrame, 
                              lgd_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Resolve block mappings for unmatched entities.
        
        Args:
            unmatched_entities: DataFrame containing entities without exact matches
            lgd_data: DataFrame containing LGD reference data
            
        Returns:
            Tuple of (resolved_entities, still_unmatched_entities)
        """
        self.logger.info("Starting block mapping resolution")
        
        if unmatched_entities.empty:
            self.logger.info("No unmatched entities to resolve")
            return pd.DataFrame(), pd.DataFrame()
        
        # Create block lookup from LGD data
        block_lookup = self._create_block_lookup(lgd_data)
        
        resolved_entities = []
        still_unmatched = []
        
        total_entities = len(unmatched_entities)
        self._resolution_stats['total_processed'] = total_entities
        
        self.logger.info(f"Processing {total_entities} unmatched entities for block resolution")
        
        with tqdm(total=total_entities, desc="Block resolution") as pbar:
            for idx, row in unmatched_entities.iterrows():
                try:
                    entity_dict = row.to_dict()
                    
                    # Try to resolve block mapping
                    resolved_mapping = self._resolve_single_block(entity_dict, block_lookup)
                    
                    if resolved_mapping:
                        # Add resolved mapping information to entity
                        entity_dict.update(resolved_mapping)
                        resolved_entities.append(entity_dict)
                        
                        method = resolved_mapping.get('resolution_method')
                        if method in ['exact', 'exact_standardized']:
                            self._resolution_stats['exact_matches'] += 1
                        elif method == 'fuzzy':
                            self._resolution_stats['fuzzy_matches'] += 1
                    else:
                        still_unmatched.append(entity_dict)
                        self._resolution_stats['unresolved'] += 1
                
                except Exception as e:
                    self.logger.warning(f"Error resolving block for entity at row {idx}: {e}")
                    still_unmatched.append(row.to_dict())
                    self._resolution_stats['unresolved'] += 1
                
                pbar.update(1)
        
        # Convert results back to DataFrames
        resolved_df = pd.DataFrame(resolved_entities) if resolved_entities else pd.DataFrame()
        unmatched_df = pd.DataFrame(still_unmatched) if still_unmatched else pd.DataFrame()
        
        # Log resolution statistics
        self._log_resolution_statistics()
        
        return resolved_df, unmatched_df
    
    def standardize_block_name(self, block_name: str) -> str:
        """
        Standardize block name for better matching.
        
        Args:
            block_name: Original block name
            
        Returns:
            Standardized block name
        """
        if is_null_or_empty(block_name):
            return ""
        
        # Check cache first
        if block_name in self._standardization_cache:
            return self._standardization_cache[block_name]
        
        # Convert to string and clean
        standardized = safe_string_conversion(block_name)
        
        # Convert to uppercase for consistency
        standardized = standardized.upper()
        
        # Remove extra whitespace
        standardized = re.sub(r'\s+', ' ', standardized).strip()
        
        # Standardize common abbreviations and variations
        standardizations = {
            # Block variations - expand abbreviations to full word
            r'\bBLK\b': 'BLOCK',
            r'\bBL\b': 'BLOCK',
            
            # Direction abbreviations
            r'\bN\b': 'NORTH',
            r'\bS\b': 'SOUTH',
            r'\bE\b': 'EAST',
            r'\bW\b': 'WEST',
            r'\bNE\b': 'NORTH EAST',
            r'\bNW\b': 'NORTH WEST',
            r'\bSE\b': 'SOUTH EAST',
            r'\bSW\b': 'SOUTH WEST',
            
            # Common word standardizations
            r'\bST\.\s*': 'SAINT ',
            r'\bST\b': 'SAINT',
            r'\bMT\.\s*': 'MOUNT ',
            r'\bMT\b': 'MOUNT',
            r'\bFT\b': 'FORT',
            
            # Standardize punctuation - do this before removing block words
            r'[^\w\s]': ' ',  # Replace punctuation with spaces
        }
        
        for pattern, replacement in standardizations.items():
            standardized = re.sub(pattern, replacement, standardized)
        
        # Clean up multiple spaces after substitutions
        standardized = re.sub(r'\s+', ' ', standardized).strip()
        
        # Cache the result
        self._standardization_cache[block_name] = standardized
        self._resolution_stats['standardization_applied'] += 1
        
        return standardized
    
    def find_best_block_match(self, entity_block: str, available_blocks: List[str], 
                             district_code: Optional[int] = None) -> Optional[Tuple[str, float, str]]:
        """
        Find the best matching block name using fuzzy matching.
        
        Args:
            entity_block: Block name from entity
            available_blocks: List of available block names from LGD data
            district_code: Optional district code to filter blocks
            
        Returns:
            Tuple of (matched_block, score, method) or None if no good match
        """
        if is_null_or_empty(entity_block) or not available_blocks:
            return None
        
        # Standardize the entity block name
        standardized_entity = self.standardize_block_name(entity_block)
        
        # Standardize available blocks
        standardized_available = [self.standardize_block_name(block) for block in available_blocks]
        
        # Try exact match first on standardized names
        for i, std_block in enumerate(standardized_available):
            if std_block == standardized_entity:
                return available_blocks[i], 100.0, 'exact_standardized'
        
        # Try fuzzy matching
        if standardized_available:
            # Use rapidfuzz to find best match
            match_result = process.extractOne(
                standardized_entity,
                standardized_available,
                scorer=fuzz.ratio
            )
            
            if match_result and match_result[1] >= self.fuzzy_threshold:
                # Find original block name corresponding to the matched standardized name
                matched_idx = standardized_available.index(match_result[0])
                return available_blocks[matched_idx], match_result[1], 'fuzzy'
        
        return None
    
    def _create_block_lookup(self, lgd_data: pd.DataFrame) -> Dict[int, Dict[str, Dict]]:
        """
        Create a lookup dictionary for blocks organized by district.
        
        Args:
            lgd_data: DataFrame containing LGD reference data
            
        Returns:
            Dictionary mapping district_code -> block_name -> block_info
        """
        block_lookup = {}
        
        # Group by district and block to get unique combinations
        block_groups = lgd_data.groupby(['district_code', 'block']).first().reset_index()
        
        for _, row in block_groups.iterrows():
            district_code = row['district_code']
            block_name = row['block']
            
            if district_code not in block_lookup:
                block_lookup[district_code] = {}
            
            block_lookup[district_code][block_name] = {
                'district_code': district_code,
                'district': row['district'],
                'block_code': row['block_code'],
                'block': block_name,
                'standardized_name': self.standardize_block_name(block_name)
            }
        
        self.logger.info(f"Created block lookup with {len(block_groups)} unique blocks "
                        f"across {len(block_lookup)} districts")
        
        return block_lookup
    
    def _resolve_single_block(self, entity_dict: Dict, block_lookup: Dict) -> Optional[Dict]:
        """
        Resolve block mapping for a single entity.
        
        Args:
            entity_dict: Dictionary containing entity information
            block_lookup: Block lookup dictionary
            
        Returns:
            Dictionary with resolved mapping information or None
        """
        district_code = entity_dict.get('district_code')
        block_name = entity_dict.get('block')
        
        if not district_code or is_null_or_empty(block_name):
            return None
        
        # Check if district exists in lookup
        if district_code not in block_lookup:
            self.logger.debug(f"District code {district_code} not found in block lookup")
            return None
        
        district_blocks = block_lookup[district_code]
        
        # Try exact match first
        if block_name in district_blocks:
            block_info = district_blocks[block_name]
            return {
                'resolved_district_code': block_info['district_code'],
                'resolved_block_code': block_info['block_code'],
                'resolved_block_name': block_info['block'],
                'resolution_method': 'exact',
                'resolution_score': 100.0
            }
        
        # Try fuzzy matching
        available_blocks = list(district_blocks.keys())
        match_result = self.find_best_block_match(block_name, available_blocks, district_code)
        
        if match_result:
            matched_block, score, method = match_result
            block_info = district_blocks[matched_block]
            
            self.logger.debug(f"Resolved block '{block_name}' -> '{matched_block}' "
                            f"(score: {score:.1f}, method: {method})")
            
            return {
                'resolved_district_code': block_info['district_code'],
                'resolved_block_code': block_info['block_code'],
                'resolved_block_name': block_info['block'],
                'original_block_name': block_name,
                'resolution_method': method,
                'resolution_score': score
            }
        
        return None
    
    def _log_resolution_statistics(self):
        """Log detailed resolution statistics."""
        stats = self._resolution_stats
        total = stats['total_processed']
        
        if total == 0:
            self.logger.info("No entities processed for block resolution")
            return
        
        self.logger.info("Block Resolution Statistics:")
        self.logger.info(f"  Total processed: {total}")
        self.logger.info(f"  Exact matches: {stats['exact_matches']} "
                        f"({(stats['exact_matches']/total)*100:.1f}%)")
        self.logger.info(f"  Fuzzy matches: {stats['fuzzy_matches']} "
                        f"({(stats['fuzzy_matches']/total)*100:.1f}%)")
        self.logger.info(f"  Still unresolved: {stats['unresolved']} "
                        f"({(stats['unresolved']/total)*100:.1f}%)")
        self.logger.info(f"  Standardizations applied: {stats['standardization_applied']}")
        
        resolved_total = stats['exact_matches'] + stats['fuzzy_matches']
        resolution_rate = (resolved_total / total) * 100 if total > 0 else 0
        self.logger.info(f"  Overall resolution rate: {resolution_rate:.1f}%")
    
    def get_resolution_statistics(self) -> Dict[str, int]:
        """
        Get resolution statistics.
        
        Returns:
            Dictionary with resolution statistics
        """
        return self._resolution_stats.copy()
    
    def clear_cache(self):
        """Clear internal caches."""
        self._block_name_cache.clear()
        self._standardization_cache.clear()
        self.logger.debug("Cleared BlockMappingResolver caches")
    
    def get_standardization_examples(self, limit: int = 10) -> List[Tuple[str, str]]:
        """
        Get examples of block name standardizations.
        
        Args:
            limit: Maximum number of examples to return
            
        Returns:
            List of (original, standardized) tuples
        """
        examples = []
        for original, standardized in list(self._standardization_cache.items())[:limit]:
            if original != standardized:  # Only show cases where standardization changed something
                examples.append((original, standardized))
        
        return examples
    
    def validate_block_resolution(self, resolved_entities: pd.DataFrame) -> Dict[str, List]:
        """
        Validate the quality of block resolutions.
        
        Args:
            resolved_entities: DataFrame with resolved block mappings
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'high_confidence': [],  # Score >= 95
            'medium_confidence': [],  # Score 85-94
            'low_confidence': [],  # Score < 85
            'method_breakdown': {}
        }
        
        if resolved_entities.empty:
            return validation_results
        
        for idx, row in resolved_entities.iterrows():
            score = row.get('resolution_score', 0)
            method = row.get('resolution_method', 'unknown')
            
            # Categorize by confidence
            if score >= 95:
                validation_results['high_confidence'].append(idx)
            elif score >= 85:
                validation_results['medium_confidence'].append(idx)
            else:
                validation_results['low_confidence'].append(idx)
            
            # Track method usage
            if method not in validation_results['method_breakdown']:
                validation_results['method_breakdown'][method] = 0
            validation_results['method_breakdown'][method] += 1
        
        return validation_results