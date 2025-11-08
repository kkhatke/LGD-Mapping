"""
Fuzzy matching strategy for LGD mapping.

This module provides the FuzzyMatcher class that performs fuzzy string matching
for village names using configurable thresholds and provides alternative match suggestions.
"""

import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple, Set
from tqdm import tqdm
from rapidfuzz import fuzz, process

from ..models import EntityRecord, LGDRecord, MappingResult
from ..utils.uid_generator import UIDGenerator
from ..utils.data_utils import is_null_or_empty


class FuzzyMatcher:
    """
    Performs fuzzy matching using configurable similarity thresholds.
    
    This class implements village-level fuzzy matching with configurable thresholds
    and provides alternative match suggestions for unmatched records.
    """
    
    def __init__(self, threshold: int = 95, logger: Optional[logging.Logger] = None):
        """
        Initialize the FuzzyMatcher.
        
        Args:
            threshold: Minimum similarity threshold for matches (0-100)
            logger: Optional logger instance for logging operations
        """
        if not 0 <= threshold <= 100:
            raise ValueError("Threshold must be between 0 and 100")
            
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        self.uid_generator = UIDGenerator()
        self._village_choices_cache = {}
        self._block_mapping_cache = {}
        
        # Performance optimization settings
        self.max_alternatives = 5
        self.min_alternative_score = 70
        self.chunk_size = 1000  # Process in chunks for large datasets
    
    def match(self, entities: pd.DataFrame, lgd_data: pd.DataFrame) -> List[MappingResult]:
        """
        Perform fuzzy matching on entities using LGD data.
        
        Args:
            entities: DataFrame containing entity records to match
            lgd_data: DataFrame containing LGD reference data
            
        Returns:
            List of MappingResult objects with fuzzy matches
        """
        self.logger.info(f"Starting fuzzy matching process with {self.threshold}% threshold")
        
        # Validate input data
        if entities.empty or lgd_data.empty:
            self.logger.warning("Empty input data provided to fuzzy matcher")
            return []
        
        # Create block mapping and village choices for fuzzy matching
        self.logger.info("Preparing fuzzy matching data structures")
        block_mapping = self._create_block_mapping(lgd_data)
        village_choices = self._create_village_choices(lgd_data)
        
        # Perform matching
        results = []
        total_entities = len(entities)
        
        self.logger.info(f"Processing {total_entities} entities for fuzzy matching")
        
        # Process in chunks for better performance with large datasets
        for chunk_start in range(0, total_entities, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_entities)
            chunk = entities.iloc[chunk_start:chunk_end]
            
            chunk_results = self._process_chunk(chunk, block_mapping, village_choices)
            results.extend(chunk_results)
            
            self.logger.debug(f"Processed chunk {chunk_start}-{chunk_end}")
        
        # Log matching statistics
        matched_count = sum(1 for r in results if r.is_matched())
        unmatched_with_alternatives = sum(1 for r in results if not r.is_matched() and r.alternative_matches)
        unmatched_without_alternatives = sum(1 for r in results if not r.is_matched() and not r.alternative_matches)
        match_rate = (matched_count / total_entities) * 100 if total_entities > 0 else 0
        
        match_type = f"fuzzy_{self.threshold}"
        self.logger.info(f"Fuzzy matching ({self.threshold}%) completed: "
                        f"{matched_count}/{total_entities} entities matched ({match_rate:.2f}%)")
        self.logger.info(f"Alternative matches: {unmatched_with_alternatives} unmatched entities have alternatives, "
                        f"{unmatched_without_alternatives} have no alternatives")
        
        return results
    
    def fuzzy_match_villages(self, unmatched_df: pd.DataFrame, lgd_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform fuzzy matching on unmatched villages and return DataFrame with results.
        
        Args:
            unmatched_df: DataFrame containing unmatched entities
            lgd_df: DataFrame containing LGD reference data
            
        Returns:
            DataFrame with fuzzy matching results
        """
        self.logger.info(f"Performing fuzzy village matching with {self.threshold}% threshold")
        
        if unmatched_df.empty:
            self.logger.info("No unmatched entities to process")
            return unmatched_df.copy()
        
        # Create village choices grouped by district and block for better performance
        village_choices = self._create_village_choices(lgd_df)
        
        # Prepare result DataFrame
        result_df = unmatched_df.copy()
        result_df['match_found'] = False
        result_df['match_type'] = 'unmatched'
        result_df['match_score'] = None
        result_df['lgd_district_code'] = None
        result_df['lgd_block_code'] = None
        result_df['lgd_village_code'] = None
        result_df['alternative_matches'] = None
        
        matched_count = 0
        
        with tqdm(total=len(unmatched_df), desc=f"Fuzzy matching ({self.threshold}%)") as pbar:
            for idx, row in unmatched_df.iterrows():
                try:
                    # Get village choices for this district-block combination
                    district_code = row.get('district_code')
                    block_name = row.get('block', '')
                    village_name = row.get('village', '')
                    
                    if is_null_or_empty(village_name) or district_code is None:
                        pbar.update(1)
                        continue
                    
                    # Get relevant village choices
                    choices_key = f"{district_code}_{block_name}"
                    choices = village_choices.get(choices_key, {})
                    
                    if not choices:
                        # Try with all villages in the district if block-specific search fails
                        district_choices = {}
                        for key, village_dict in village_choices.items():
                            if key.startswith(f"{district_code}_"):
                                district_choices.update(village_dict)
                        choices = district_choices
                    
                    if choices:
                        # Perform fuzzy matching
                        match_result = self._find_best_fuzzy_match(village_name, choices)
                        
                        if match_result and match_result['score'] >= self.threshold:
                            lgd_record = match_result['lgd_record']
                            result_df.at[idx, 'match_found'] = True
                            result_df.at[idx, 'match_type'] = f'fuzzy_{self.threshold}'
                            result_df.at[idx, 'match_score'] = match_result['score']
                            result_df.at[idx, 'lgd_district_code'] = lgd_record.district_code
                            result_df.at[idx, 'lgd_block_code'] = lgd_record.block_code
                            result_df.at[idx, 'lgd_village_code'] = lgd_record.village_code
                            matched_count += 1
                        
                        # Find alternative matches for unmatched records
                        if not result_df.at[idx, 'match_found']:
                            alternatives = self.find_alternative_matches(village_name, list(choices.keys()))
                            if alternatives:
                                result_df.at[idx, 'alternative_matches'] = '; '.join(alternatives)
                                self.logger.debug(f"Unmatched '{village_name}' has {len(alternatives)} alternatives: {alternatives}")
                            else:
                                self.logger.debug(f"No alternatives found for unmatched '{village_name}'")
                
                except Exception as e:
                    self.logger.warning(f"Error processing fuzzy match at row {idx}: {e}")
                
                pbar.update(1)
        
        match_rate = (matched_count / len(unmatched_df)) * 100 if len(unmatched_df) > 0 else 0
        self.logger.info(f"Fuzzy matching completed: {matched_count}/{len(unmatched_df)} "
                        f"matches found ({match_rate:.2f}%)")
        
        return result_df
    
    def find_alternative_matches(self, village_name: str, lgd_villages: List[str]) -> List[str]:
        """
        Find alternative village matches with scores above minimum threshold.
        
        Args:
            village_name: Name of village to find alternatives for
            lgd_villages: List of LGD village names to search in
            
        Returns:
            List of alternative village names sorted by similarity score
        """
        if is_null_or_empty(village_name) or not lgd_villages:
            self.logger.debug(f"No alternatives search for '{village_name}': "
                            f"empty input (villages: {len(lgd_villages) if lgd_villages else 0})")
            return []
        
        try:
            # Use rapidfuzz to find top alternatives
            alternatives = process.extract(
                village_name,
                lgd_villages,
                scorer=fuzz.WRatio,
                limit=self.max_alternatives,
                score_cutoff=self.min_alternative_score
            )
            
            # Extract village names with scores for logging
            alternative_details = [(match[0], match[1]) for match in alternatives]
            alternative_names = [match[0] for match in alternatives]
            
            # Log detailed alternative match information
            if alternative_details:
                self.logger.debug(f"Found {len(alternative_details)} alternatives for '{village_name}':")
                for name, score in alternative_details:
                    confidence = self._get_match_confidence_from_score(score)
                    self.logger.debug(f"  - '{name}' (score: {score:.1f}, confidence: {confidence})")
            else:
                self.logger.debug(f"No alternatives found for '{village_name}' "
                                f"(min_score: {self.min_alternative_score}, "
                                f"searched: {len(lgd_villages)} villages)")
            
            return alternative_names
            
        except Exception as e:
            self.logger.warning(f"Error finding alternatives for '{village_name}': {e}")
            return []
    
    def _process_chunk(self, chunk: pd.DataFrame, block_mapping: Dict[str, Dict], 
                      village_choices: Dict[str, Dict[str, LGDRecord]]) -> List[MappingResult]:
        """
        Process a chunk of entities for fuzzy matching.
        
        Args:
            chunk: DataFrame chunk to process
            block_mapping: Block-level mapping lookup
            village_choices: Village choices for fuzzy matching
            
        Returns:
            List of MappingResult objects for the chunk
        """
        results = []
        
        with tqdm(total=len(chunk), desc=f"Fuzzy chunk ({self.threshold}%)", leave=False) as pbar:
            for idx, row in chunk.iterrows():
                try:
                    # Create EntityRecord
                    entity = EntityRecord(
                        district=row['district'],
                        block=row['block'],
                        village=row['village'],
                        district_code=row.get('district_code')
                    )
                    
                    # Skip invalid entities
                    if not entity.is_valid():
                        results.append(MappingResult(
                            entity=entity,
                            lgd_match=None,
                            match_type='unmatched'
                        ))
                        pbar.update(1)
                        continue
                    
                    # Attempt fuzzy matching
                    lgd_match, match_score, alternatives = self._find_fuzzy_match(
                        entity, block_mapping, village_choices
                    )
                    
                    if lgd_match and match_score >= self.threshold:
                        match_type = f"fuzzy_{self.threshold}"
                        results.append(MappingResult(
                            entity=entity,
                            lgd_match=lgd_match,
                            match_type=match_type,
                            match_score=match_score / 100.0,  # Convert to 0-1 scale
                            alternative_matches=alternatives
                        ))
                    else:
                        results.append(MappingResult(
                            entity=entity,
                            lgd_match=None,
                            match_type='unmatched',
                            alternative_matches=alternatives
                        ))
                
                except Exception as e:
                    self.logger.warning(f"Error processing entity at row {idx}: {e}")
                    entity = EntityRecord(
                        district=str(row.get('district', '')),
                        block=str(row.get('block', '')),
                        village=str(row.get('village', '')),
                        district_code=row.get('district_code')
                    )
                    results.append(MappingResult(
                        entity=entity,
                        lgd_match=None,
                        match_type='unmatched'
                    ))
                
                pbar.update(1)
        
        return results
    
    def _create_block_mapping(self, lgd_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Create block-level mapping lookup for district code resolution.
        
        Args:
            lgd_data: DataFrame containing LGD reference data
            
        Returns:
            Dictionary mapping block UIDs to block information
        """
        if self._block_mapping_cache:
            return self._block_mapping_cache
        
        block_mapping = {}
        
        # Group by district_code and block to get unique combinations
        block_groups = lgd_data.groupby(['district_code', 'block']).first().reset_index()
        
        for _, row in block_groups.iterrows():
            district_code = row['district_code']
            block_name = row['block']
            
            # Create multiple keys for flexible matching
            keys = [
                f"{district_code}_{block_name}",
                f"{district_code}_{block_name.lower()}",
                f"{district_code}_{block_name.strip()}"
            ]
            
            block_info = {
                'district_code': district_code,
                'district': row['district'],
                'block_code': row['block_code'],
                'block': block_name
            }
            
            for key in keys:
                if key not in block_mapping:
                    block_mapping[key] = block_info
        
        self.logger.info(f"Created block mapping with {len(block_mapping)} entries")
        self._block_mapping_cache = block_mapping
        return block_mapping
    
    def _create_village_choices(self, lgd_data: pd.DataFrame) -> Dict[str, Dict[str, LGDRecord]]:
        """
        Create village choices grouped by district and block for efficient fuzzy matching.
        
        Args:
            lgd_data: DataFrame containing LGD reference data
            
        Returns:
            Dictionary mapping district_block keys to village choices
        """
        if self._village_choices_cache:
            return self._village_choices_cache
        
        village_choices = {}
        
        for _, row in lgd_data.iterrows():
            try:
                # Create LGDRecord
                lgd_record = LGDRecord(
                    district_code=row['district_code'],
                    district=row['district'],
                    block_code=row['block_code'],
                    block=row['block'],
                    village_code=row['village_code'],
                    village=row['village'],
                    gp_code=row.get('gp_code'),
                    gp=row.get('gp')
                )
                
                # Skip invalid records
                if not lgd_record.is_valid():
                    continue
                
                # Create key for grouping
                key = f"{lgd_record.district_code}_{lgd_record.block}"
                
                if key not in village_choices:
                    village_choices[key] = {}
                
                village_choices[key][lgd_record.village] = lgd_record
                
            except Exception as e:
                self.logger.debug(f"Error creating village choice: {e}")
                continue
        
        total_villages = sum(len(villages) for villages in village_choices.values())
        self.logger.info(f"Created village choices with {len(village_choices)} district-block "
                        f"combinations and {total_villages} total villages")
        
        self._village_choices_cache = village_choices
        return village_choices
    
    def _find_fuzzy_match(self, entity: EntityRecord, block_mapping: Dict[str, Dict],
                         village_choices: Dict[str, Dict[str, LGDRecord]]) -> Tuple[Optional[LGDRecord], float, List[str]]:
        """
        Find fuzzy match for an entity.
        
        Args:
            entity: EntityRecord to match
            block_mapping: Block-level mapping lookup
            village_choices: Village choices for fuzzy matching
            
        Returns:
            Tuple of (LGDRecord if match found, match score, alternative matches)
        """
        alternatives = []
        
        # Ensure we have district_code
        district_code = entity.district_code
        if district_code is None:
            # Try to resolve district_code through block mapping
            # This is a simplified approach - in practice, you might need more sophisticated resolution
            return None, 0.0, alternatives
        
        # Get village choices for this district-block combination
        choices_key = f"{district_code}_{entity.block}"
        choices = village_choices.get(choices_key, {})
        
        if not choices:
            # Try with all villages in the district if block-specific search fails
            district_choices = {}
            for key, village_dict in village_choices.items():
                if key.startswith(f"{district_code}_"):
                    district_choices.update(village_dict)
            choices = district_choices
        
        if not choices:
            return None, 0.0, alternatives
        
        # Find best fuzzy match
        match_result = self._find_best_fuzzy_match(entity.village, choices)
        
        if match_result:
            # Find alternatives regardless of whether we have a match above threshold
            alternatives = self.find_alternative_matches(entity.village, list(choices.keys()))
            return match_result['lgd_record'], match_result['score'], alternatives
        
        return None, 0.0, alternatives
    
    def _find_best_fuzzy_match(self, village_name: str, choices: Dict[str, LGDRecord]) -> Optional[Dict]:
        """
        Find the best fuzzy match from available choices.
        
        Args:
            village_name: Village name to match
            choices: Dictionary of village names to LGDRecord objects
            
        Returns:
            Dictionary with best match information or None
        """
        if is_null_or_empty(village_name) or not choices:
            return None
        
        try:
            # Use rapidfuzz to find the best match
            best_match = process.extractOne(
                village_name,
                list(choices.keys()),
                scorer=fuzz.WRatio,
                score_cutoff=self.min_alternative_score
            )
            
            if best_match:
                matched_village_name, score, _ = best_match  # rapidfuzz returns (match, score, index)
                lgd_record = choices[matched_village_name]
                
                return {
                    'lgd_record': lgd_record,
                    'score': score,
                    'matched_name': matched_village_name
                }
            
        except Exception as e:
            self.logger.warning(f"Error in fuzzy matching for '{village_name}': {e}")
        
        return None
    
    def get_strategy_name(self) -> str:
        """
        Get the name of this matching strategy.
        
        Returns:
            Strategy name string
        """
        return f"fuzzy_{self.threshold}"
    
    def get_matching_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the matching process.
        
        Returns:
            Dictionary with matching statistics
        """
        return {
            'threshold': self.threshold,
            'village_choices_created': len(self._village_choices_cache),
            'block_mappings_created': len(self._block_mapping_cache),
            'max_alternatives': self.max_alternatives,
            'min_alternative_score': self.min_alternative_score
        }
    
    def clear_cache(self):
        """Clear internal caches."""
        self._village_choices_cache.clear()
        self._block_mapping_cache.clear()
        self.uid_generator.clear_registry()
        self.logger.debug("Cleared FuzzyMatcher caches")
    
    def _get_match_confidence_from_score(self, score: float) -> str:
        """
        Get confidence level from match score.
        
        Args:
            score: Match score (0-100)
            
        Returns:
            Confidence level string
        """
        if score >= 95:
            return 'High'
        elif score >= 85:
            return 'Medium'
        else:
            return 'Low'
    
    def set_performance_options(self, chunk_size: int = None, max_alternatives: int = None,
                               min_alternative_score: int = None):
        """
        Set performance optimization options.
        
        Args:
            chunk_size: Size of chunks for processing large datasets
            max_alternatives: Maximum number of alternative matches to find
            min_alternative_score: Minimum score for alternative matches
        """
        if chunk_size is not None:
            self.chunk_size = max(1, chunk_size)
        if max_alternatives is not None:
            self.max_alternatives = max(1, max_alternatives)
        if min_alternative_score is not None:
            self.min_alternative_score = max(0, min(100, min_alternative_score))
        
        self.logger.info(f"Updated performance options: chunk_size={self.chunk_size}, "
                        f"max_alternatives={self.max_alternatives}, "
                        f"min_alternative_score={self.min_alternative_score}")