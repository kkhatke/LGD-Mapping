"""
Fuzzy matching strategy for LGD mapping.

This module provides the FuzzyMatcher class that performs fuzzy string matching
for village names using configurable thresholds and provides alternative match suggestions.
Enhanced with hierarchical matching support across all administrative levels.
"""

import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple, Set, TYPE_CHECKING
from tqdm import tqdm
from rapidfuzz import fuzz, process

from ..models import EntityRecord, LGDRecord, MappingResult
from ..utils.uid_generator import UIDGenerator
from ..utils.data_utils import is_null_or_empty

if TYPE_CHECKING:
    from ..hierarchy.hierarchy_config import HierarchyConfiguration


class FuzzyMatcher:
    """
    Performs fuzzy matching using configurable similarity thresholds.
    
    This class implements hierarchical fuzzy matching across all administrative levels
    with level-specific thresholds and provides alternative match suggestions.
    """
    
    def __init__(
        self, 
        threshold: int = 95, 
        logger: Optional[logging.Logger] = None,
        hierarchy_config: Optional['HierarchyConfiguration'] = None,
        enable_progressive_filtering: bool = True,
        enable_batch_processing: bool = True
    ):
        """
        Initialize the FuzzyMatcher with hierarchical support and performance optimizations.
        
        Args:
            threshold: Default minimum similarity threshold for matches (0-100)
            logger: Optional logger instance for logging operations
            hierarchy_config: Optional hierarchy configuration for level-aware matching
            enable_progressive_filtering: Enable progressive filtering by parent codes
            enable_batch_processing: Enable batch processing by parent codes
        """
        if not 0 <= threshold <= 100:
            raise ValueError("Threshold must be between 0 and 100")
            
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        self.hierarchy_config = hierarchy_config
        self.uid_generator = UIDGenerator()
        self.enable_progressive_filtering = enable_progressive_filtering
        self.enable_batch_processing = enable_batch_processing
        self._village_choices_cache = {}
        self._block_mapping_cache = {}
        self._hierarchical_choices_cache = {}
        self._parent_filtered_cache = {}  # Cache for parent-filtered choices
        
        # Performance optimization settings
        self.max_alternatives = 5
        self.min_alternative_score = 70
        self.chunk_size = 1000  # Process in chunks for large datasets
        self._cache_hits = 0
        self._cache_misses = 0
    
    def match(self, entities: pd.DataFrame, lgd_data: pd.DataFrame) -> List[MappingResult]:
        """
        Perform hierarchical fuzzy matching on entities using LGD data.
        
        Uses level-aware fuzzy matching with hierarchy configuration to match
        entities across all administrative levels with level-specific thresholds.
        
        Args:
            entities: DataFrame containing entity records to match
            lgd_data: DataFrame containing LGD reference data
            
        Returns:
            List of MappingResult objects with fuzzy matches and hierarchy details
        """
        self.logger.info(f"Starting hierarchical fuzzy matching process with {self.threshold}% default threshold")
        
        # Validate input data
        if entities.empty or lgd_data.empty:
            self.logger.warning("Empty input data provided to fuzzy matcher")
            return []
        
        # Log hierarchy configuration if available
        if self.hierarchy_config:
            detected_levels = self.hierarchy_config.detected_levels
            self.logger.info(f"Using hierarchical matching with levels: {detected_levels}")
            
            # Log level-specific thresholds
            for level in detected_levels:
                threshold = self.hierarchy_config.get_fuzzy_threshold(level)
                if threshold:
                    self.logger.info(f"  {level}: {threshold}% threshold")
        else:
            self.logger.info("No hierarchy configuration provided, using basic fuzzy matching")
        
        # Create matching data structures
        self.logger.info("Preparing fuzzy matching data structures")
        block_mapping = self._create_block_mapping(lgd_data)
        village_choices = self._create_village_choices(lgd_data)
        
        # Create hierarchical choices if hierarchy config is available
        if self.hierarchy_config:
            hierarchical_choices = self._create_hierarchical_choices(lgd_data)
        else:
            hierarchical_choices = {}
        
        # Perform matching
        results = []
        total_entities = len(entities)
        
        self.logger.info(f"Processing {total_entities} entities for fuzzy matching")
        
        # Process in chunks or batches for better performance with large datasets
        if self.enable_batch_processing and self.hierarchy_config and total_entities >= self.chunk_size:
            results = self._process_entities_in_batches_by_parent(
                entities, block_mapping, village_choices, hierarchical_choices
            )
        else:
            # Process in chunks for better performance with large datasets
            for chunk_start in range(0, total_entities, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, total_entities)
                chunk = entities.iloc[chunk_start:chunk_end]
                
                chunk_results = self._process_chunk(
                    chunk, block_mapping, village_choices, hierarchical_choices
                )
                results.extend(chunk_results)
                
                self.logger.debug(f"Processed chunk {chunk_start}-{chunk_end}")
        
        # Log matching statistics
        matched_count = sum(1 for r in results if r.is_matched())
        unmatched_with_alternatives = sum(1 for r in results if not r.is_matched() and r.alternative_matches)
        unmatched_without_alternatives = sum(1 for r in results if not r.is_matched() and not r.alternative_matches)
        match_rate = (matched_count / total_entities) * 100 if total_entities > 0 else 0
        
        # Log hierarchical match statistics if available
        if self.hierarchy_config and matched_count > 0:
            self._log_hierarchical_statistics(results)
        
        match_type = f"fuzzy_{self.threshold}"
        
        # Log with hierarchical context
        hierarchy_depth = self.hierarchy_config.get_hierarchy_depth() if self.hierarchy_config else 3
        detected_levels = self.hierarchy_config.detected_levels if self.hierarchy_config else ['district', 'block', 'village']
        
        self.logger.info(
            f"Fuzzy matching ({self.threshold}%) completed: {matched_count}/{total_entities} matched ({match_rate:.2f}%)",
            extra={
                'threshold': self.threshold,
                'total_entities': total_entities,
                'matched_count': matched_count,
                'match_rate': match_rate,
                'unmatched_with_alternatives': unmatched_with_alternatives,
                'unmatched_without_alternatives': unmatched_without_alternatives,
                'hierarchy_depth': hierarchy_depth,
                'detected_levels': detected_levels
            }
        )
        
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
    
    def _process_chunk(
        self, 
        chunk: pd.DataFrame, 
        block_mapping: Dict[str, Dict], 
        village_choices: Dict[str, Dict[str, LGDRecord]],
        hierarchical_choices: Dict[str, Dict[str, LGDRecord]] = None
    ) -> List[MappingResult]:
        """
        Process a chunk of entities for hierarchical fuzzy matching.
        
        Args:
            chunk: DataFrame chunk to process
            block_mapping: Block-level mapping lookup
            village_choices: Village choices for fuzzy matching
            hierarchical_choices: Hierarchical choices for level-aware matching
            
        Returns:
            List of MappingResult objects for the chunk with hierarchy details
        """
        results = []
        
        with tqdm(total=len(chunk), desc=f"Fuzzy chunk ({self.threshold}%)", leave=False) as pbar:
            for idx, row in chunk.iterrows():
                try:
                    # Create EntityRecord with hierarchical fields
                    entity = EntityRecord(
                        district=row['district'],
                        block=row['block'],
                        village=row['village'],
                        state=row.get('state'),
                        gp=row.get('gp'),
                        subdistrict=row.get('subdistrict'),
                        state_code=row.get('state_code'),
                        district_code=row.get('district_code'),
                        block_code=row.get('block_code'),
                        gp_code=row.get('gp_code'),
                        village_code=row.get('village_code')
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
                    
                    # Attempt hierarchical fuzzy matching
                    if self.hierarchy_config and hierarchical_choices:
                        lgd_match, match_score, alternatives, hierarchy_details = self._find_hierarchical_fuzzy_match(
                            entity, hierarchical_choices
                        )
                    else:
                        # Fall back to basic fuzzy matching
                        lgd_match, match_score, alternatives = self._find_fuzzy_match(
                            entity, block_mapping, village_choices
                        )
                        hierarchy_details = {}
                    
                    if lgd_match and match_score >= self.threshold:
                        match_type = f"fuzzy_{self.threshold}"
                        results.append(MappingResult(
                            entity=entity,
                            lgd_match=lgd_match,
                            match_type=match_type,
                            match_score=match_score / 100.0,  # Convert to 0-1 scale
                            alternative_matches=alternatives,
                            hierarchy_match_details=hierarchy_details
                        ))
                    else:
                        results.append(MappingResult(
                            entity=entity,
                            lgd_match=None,
                            match_type='unmatched',
                            alternative_matches=alternatives,
                            hierarchy_match_details=hierarchy_details
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
    
    def _create_hierarchical_choices(
        self, 
        lgd_data: pd.DataFrame
    ) -> Dict[str, Dict[str, LGDRecord]]:
        """
        Create hierarchical choices for level-aware fuzzy matching with progressive filtering.
        
        Groups LGD records by parent codes to enable scoped fuzzy matching
        at each hierarchical level. Uses progressive filtering to reduce search space.
        
        Args:
            lgd_data: DataFrame containing LGD reference data
            
        Returns:
            Dictionary mapping parent context keys to LGD records
        """
        if self._hierarchical_choices_cache:
            self.logger.debug("Using cached hierarchical choices")
            return self._hierarchical_choices_cache
        
        if not self.hierarchy_config:
            return {}
        
        self.logger.info("Creating hierarchical choices with progressive filtering")
        hierarchical_choices = {}
        
        for _, row in lgd_data.iterrows():
            try:
                # Create LGDRecord with all hierarchical fields
                lgd_record = LGDRecord(
                    district_code=row['district_code'],
                    district=row['district'],
                    block_code=row['block_code'],
                    block=row['block'],
                    village_code=row['village_code'],
                    village=row['village'],
                    state_code=row.get('state_code'),
                    state=row.get('state'),
                    gp_code=row.get('gp_code'),
                    gp=row.get('gp'),
                    subdistrict_code=row.get('subdistrict_code'),
                    subdistrict=row.get('subdistrict')
                )
                
                # Skip invalid records
                if not lgd_record.is_valid():
                    continue
                
                # Create hierarchical keys based on detected levels
                # For each level, create a key with parent context
                for level in self.hierarchy_config.detected_levels:
                    parent_key = self._build_parent_context_key(lgd_record, level)
                    level_name = self._get_level_name_value(lgd_record, level)
                    
                    if parent_key and level_name:
                        full_key = f"{parent_key}_{level}"
                        
                        if full_key not in hierarchical_choices:
                            hierarchical_choices[full_key] = {}
                        
                        hierarchical_choices[full_key][level_name] = lgd_record
                
            except Exception as e:
                self.logger.debug(f"Error creating hierarchical choice: {e}")
                continue
        
        total_entries = sum(len(choices) for choices in hierarchical_choices.values())
        self.logger.info(f"Created hierarchical choices with {len(hierarchical_choices)} "
                        f"context groups and {total_entries} total entries")
        
        self._hierarchical_choices_cache = hierarchical_choices
        return hierarchical_choices
    
    def _build_parent_context_key(self, lgd_record: LGDRecord, level: str) -> Optional[str]:
        """
        Build parent context key for a given level.
        
        Args:
            lgd_record: LGD record to extract parent context from
            level: Level name to build context for
            
        Returns:
            Parent context key string or None
        """
        if not self.hierarchy_config:
            return None
        
        # Get parent level
        parent_level = None
        for i, detected_level in enumerate(self.hierarchy_config.detected_levels):
            if detected_level == level and i > 0:
                parent_level = self.hierarchy_config.detected_levels[i - 1]
                break
        
        if not parent_level:
            return None
        
        # Build key from parent codes
        parent_codes = []
        for detected_level in self.hierarchy_config.detected_levels:
            if detected_level == level:
                break
            
            code = self._get_level_code_value(lgd_record, detected_level)
            if code is not None:
                parent_codes.append(str(code))
        
        return '_'.join(parent_codes) if parent_codes else None
    
    def _get_level_name_value(self, lgd_record: LGDRecord, level: str) -> Optional[str]:
        """
        Get name value for a specific level from LGD record.
        
        Args:
            lgd_record: LGD record to extract from
            level: Level name
            
        Returns:
            Name value or None
        """
        level_map = {
            'state': lgd_record.state,
            'district': lgd_record.district,
            'block': lgd_record.block,
            'subdistrict': lgd_record.subdistrict,
            'gp': lgd_record.gp,
            'village': lgd_record.village
        }
        return level_map.get(level)
    
    def _get_level_code_value(self, lgd_record: LGDRecord, level: str) -> Optional[int]:
        """
        Get code value for a specific level from LGD record.
        
        Args:
            lgd_record: LGD record to extract from
            level: Level name
            
        Returns:
            Code value or None
        """
        level_map = {
            'state': lgd_record.state_code,
            'district': lgd_record.district_code,
            'block': lgd_record.block_code,
            'subdistrict': lgd_record.subdistrict_code,
            'gp': lgd_record.gp_code,
            'village': lgd_record.village_code
        }
        return level_map.get(level)
    
    def _find_hierarchical_fuzzy_match(
        self,
        entity: EntityRecord,
        hierarchical_choices: Dict[str, Dict[str, LGDRecord]]
    ) -> Tuple[Optional[LGDRecord], float, List[str], Dict[str, str]]:
        """
        Find hierarchical fuzzy match using level-aware matching with progressive filtering.
        
        Performs fuzzy matching at the lowest available level with parent context,
        using level-specific thresholds from hierarchy configuration and caching.
        
        Args:
            entity: EntityRecord to match
            hierarchical_choices: Hierarchical choices for matching
            
        Returns:
            Tuple of (LGDRecord if match found, match score, alternatives, hierarchy details)
        """
        hierarchy_details = {}
        alternatives = []
        
        if not self.hierarchy_config:
            return None, 0.0, alternatives, hierarchy_details
        
        # Get the lowest level to match (typically village)
        detected_levels = self.hierarchy_config.detected_levels
        if not detected_levels:
            return None, 0.0, alternatives, hierarchy_details
        
        target_level = detected_levels[-1]  # Lowest level (e.g., village)
        
        # Build parent context key from entity
        parent_key = self._build_entity_parent_context_key(entity, target_level)
        
        if not parent_key:
            # Fall back to basic matching if we can't build parent context
            return None, 0.0, alternatives, hierarchy_details
        
        # Get choices for this parent context with caching
        full_key = f"{parent_key}_{target_level}"
        
        # Check cache first
        if self.enable_progressive_filtering and full_key in self._parent_filtered_cache:
            choices = self._parent_filtered_cache[full_key]
            self._cache_hits += 1
        else:
            choices = hierarchical_choices.get(full_key, {})
            self._cache_misses += 1
            
            # Cache the choices if progressive filtering is enabled
            if self.enable_progressive_filtering and choices:
                self._parent_filtered_cache[full_key] = choices
                self._manage_cache_size()
        
        if not choices:
            # Try broader context if exact parent context has no matches
            # For example, try district-level context if block-level has no matches
            for i in range(len(detected_levels) - 2, -1, -1):
                broader_level = detected_levels[i]
                broader_key = self._build_entity_parent_context_key(entity, broader_level)
                if broader_key:
                    full_key = f"{broader_key}_{target_level}"
                    choices = hierarchical_choices.get(full_key, {})
                    if choices:
                        self.logger.debug(f"Using broader context at {broader_level} level")
                        break
        
        if not choices:
            return None, 0.0, alternatives, hierarchy_details
        
        # Get target name from entity
        target_name = self._get_entity_level_name(entity, target_level)
        
        if not target_name:
            return None, 0.0, alternatives, hierarchy_details
        
        # Get level-specific threshold
        level_threshold = self.hierarchy_config.get_fuzzy_threshold(target_level)
        if level_threshold is None:
            level_threshold = self.threshold
        
        # Perform fuzzy matching
        match_result = self._find_best_fuzzy_match(target_name, choices)
        
        if match_result:
            # Build hierarchy match details
            lgd_record = match_result['lgd_record']
            
            for level in detected_levels:
                entity_name = self._get_entity_level_name(entity, level)
                lgd_name = self._get_level_name_value(lgd_record, level)
                
                if entity_name and lgd_name:
                    if entity_name.lower() == lgd_name.lower():
                        hierarchy_details[level] = 'exact'
                    else:
                        # Calculate fuzzy score for this level
                        level_score = fuzz.WRatio(entity_name, lgd_name)
                        hierarchy_details[level] = f'fuzzy_{int(level_score)}'
                else:
                    hierarchy_details[level] = 'missing'
            
            # Find alternatives
            alternatives = self.find_alternative_matches(target_name, list(choices.keys()))
            
            return lgd_record, match_result['score'], alternatives, hierarchy_details
        
        return None, 0.0, alternatives, hierarchy_details
    
    def _build_entity_parent_context_key(self, entity: EntityRecord, level: str) -> Optional[str]:
        """
        Build parent context key from entity for a given level.
        
        Args:
            entity: EntityRecord to extract parent context from
            level: Level name to build context for
            
        Returns:
            Parent context key string or None
        """
        if not self.hierarchy_config:
            return None
        
        # Get parent level
        parent_level = None
        for i, detected_level in enumerate(self.hierarchy_config.detected_levels):
            if detected_level == level and i > 0:
                parent_level = self.hierarchy_config.detected_levels[i - 1]
                break
        
        if not parent_level:
            return None
        
        # Build key from parent codes
        parent_codes = []
        for detected_level in self.hierarchy_config.detected_levels:
            if detected_level == level:
                break
            
            code = self._get_entity_level_code(entity, detected_level)
            if code is not None:
                parent_codes.append(str(code))
        
        return '_'.join(parent_codes) if parent_codes else None
    
    def _get_entity_level_name(self, entity: EntityRecord, level: str) -> Optional[str]:
        """
        Get name value for a specific level from entity record.
        
        Args:
            entity: EntityRecord to extract from
            level: Level name
            
        Returns:
            Name value or None
        """
        level_map = {
            'state': entity.state,
            'district': entity.district,
            'block': entity.block,
            'subdistrict': entity.subdistrict,
            'gp': entity.gp,
            'village': entity.village
        }
        return level_map.get(level)
    
    def _get_entity_level_code(self, entity: EntityRecord, level: str) -> Optional[int]:
        """
        Get code value for a specific level from entity record.
        
        Args:
            entity: EntityRecord to extract from
            level: Level name
            
        Returns:
            Code value or None
        """
        level_map = {
            'state': entity.state_code,
            'district': entity.district_code,
            'block': entity.block_code,
            'subdistrict': entity.subdistrict_code,
            'gp': entity.gp_code,
            'village': entity.village_code
        }
        return level_map.get(level)
    
    def _log_hierarchical_statistics(self, results: List[MappingResult]):
        """
        Log hierarchical matching statistics.
        
        Args:
            results: List of mapping results to analyze
        """
        if not self.hierarchy_config:
            return
        
        # Count matches by hierarchy level
        level_stats = {}
        
        for level in self.hierarchy_config.detected_levels:
            exact_count = 0
            fuzzy_count = 0
            missing_count = 0
            
            for result in results:
                if result.is_matched() and result.hierarchy_match_details:
                    match_type = result.hierarchy_match_details.get(level, 'missing')
                    
                    if match_type == 'exact':
                        exact_count += 1
                    elif match_type.startswith('fuzzy_'):
                        fuzzy_count += 1
                    else:
                        missing_count += 1
            
            level_stats[level] = {
                'exact': exact_count,
                'fuzzy': fuzzy_count,
                'missing': missing_count
            }
        
        # Log statistics
        self.logger.info("Hierarchical matching statistics:")
        for level, stats in level_stats.items():
            total = stats['exact'] + stats['fuzzy'] + stats['missing']
            if total > 0:
                exact_pct = (stats['exact'] / total) * 100
                fuzzy_pct = (stats['fuzzy'] / total) * 100
                self.logger.info(f"  {level}: {stats['exact']} exact ({exact_pct:.1f}%), "
                               f"{stats['fuzzy']} fuzzy ({fuzzy_pct:.1f}%), "
                               f"{stats['missing']} missing")
    
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
    
    def clear_cache(self):
        """
        Clear all internal caches to free memory.
        
        This method should be called after processing is complete or when
        switching to a new dataset to free up memory used by caches.
        """
        self._village_choices_cache.clear()
        self._block_mapping_cache.clear()
        self._hierarchical_choices_cache.clear()
        self._parent_filtered_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger.debug("Cleared all fuzzy matcher caches")
    
    def get_cache_statistics(self) -> Dict[str, any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_lookups = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_lookups * 100) if total_lookups > 0 else 0
        
        # Estimate cache size
        cache_size_bytes = (
            len(self._village_choices_cache) * 500 +
            len(self._block_mapping_cache) * 200 +
            len(self._hierarchical_choices_cache) * 500 +
            len(self._parent_filtered_cache) * 300
        )
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': hit_rate,
            'cache_size_mb': cache_size_bytes / (1024 * 1024),
            'village_choices_entries': len(self._village_choices_cache),
            'block_mapping_entries': len(self._block_mapping_cache),
            'hierarchical_choices_entries': len(self._hierarchical_choices_cache),
            'parent_filtered_entries': len(self._parent_filtered_cache),
            'progressive_filtering_enabled': self.enable_progressive_filtering,
            'batch_processing_enabled': self.enable_batch_processing
        }
    def _manage_cache_size(self):
        """
        Manage cache size to prevent excessive memory usage.
        
        Clears parent filtered cache if it grows too large.
        """
        # Limit parent filtered cache to 10000 entries
        if len(self._parent_filtered_cache) > 10000:
            self.logger.warning(
                f"Parent filtered cache size ({len(self._parent_filtered_cache)}) "
                f"exceeds limit, clearing cache"
            )
            self._parent_filtered_cache.clear()
    
    def _process_entities_in_batches_by_parent(
        self,
        entities: pd.DataFrame,
        block_mapping: Dict[str, Dict],
        village_choices: Dict[str, Dict[str, LGDRecord]],
        hierarchical_choices: Dict[str, Dict[str, LGDRecord]]
    ) -> List[MappingResult]:
        """
        Process entities in batches grouped by parent codes for better performance.
        
        Groups entities by their parent codes and processes each group together,
        improving cache efficiency and reducing redundant lookups.
        
        Args:
            entities: Entity DataFrame
            block_mapping: Block-level mapping
            village_choices: Village choices for matching
            hierarchical_choices: Hierarchical choices for matching
            
        Returns:
            List of MappingResult objects
        """
        if not self.enable_batch_processing or not self.hierarchy_config:
            # Fall back to regular chunk processing
            results = []
            for chunk_start in range(0, len(entities), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(entities))
                chunk = entities.iloc[chunk_start:chunk_end]
                chunk_results = self._process_chunk(
                    chunk, block_mapping, village_choices, hierarchical_choices
                )
                results.extend(chunk_results)
            return results
        
        self.logger.info("Processing entities in batches by parent codes")
        
        # Get grouping level (typically district)
        detected_levels = self.hierarchy_config.detected_levels
        if len(detected_levels) < 2:
            # Not enough levels for batching
            results = []
            for chunk_start in range(0, len(entities), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(entities))
                chunk = entities.iloc[chunk_start:chunk_end]
                chunk_results = self._process_chunk(
                    chunk, block_mapping, village_choices, hierarchical_choices
                )
                results.extend(chunk_results)
            return results
        
        grouping_level = detected_levels[0]
        grouping_code_col = self.hierarchy_config.get_level(grouping_level).code_column
        
        if grouping_code_col not in entities.columns or entities[grouping_code_col].isna().all():
            # Can't group (column missing or all NaN), fall back to regular processing
            self.logger.debug(f"Cannot group by {grouping_code_col} (missing or all NaN), using regular processing")
            results = []
            for chunk_start in range(0, len(entities), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(entities))
                chunk = entities.iloc[chunk_start:chunk_end]
                chunk_results = self._process_chunk(
                    chunk, block_mapping, village_choices, hierarchical_choices
                )
                results.extend(chunk_results)
            return results
        
        # Group entities by parent code (excluding NaN values)
        grouped = entities[entities[grouping_code_col].notna()].groupby(grouping_code_col)
        total_groups = len(grouped)
        
        self.logger.info(f"Processing {total_groups} batches grouped by {grouping_level}")
        
        results = []
        for parent_code, group_df in grouped:
            # Process this batch
            batch_results = self._process_chunk(
                group_df, block_mapping, village_choices, hierarchical_choices
            )
            results.extend(batch_results)
        
        return results
