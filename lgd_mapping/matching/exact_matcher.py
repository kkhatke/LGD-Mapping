"""
Exact matching strategy for LGD mapping.

This module provides the ExactMatcher class that performs direct UID-based matching
for both block-level and village-level entities.
"""

import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple, Set, Any
from tqdm import tqdm

from ..models import EntityRecord, LGDRecord, MappingResult
from ..exceptions import MatchingError, ValidationError, MappingProcessError
from ..utils.uid_generator import UIDGenerator
from ..utils.data_utils import is_null_or_empty
from ..utils.error_handler import (
    ErrorHandler, with_error_handling, create_error_context, log_error_details
)


class ExactMatcher:
    """
    Performs exact matching using unique identifiers (UIDs) for administrative entities.
    
    This class implements both block-level and village-level exact matching strategies
    using generated UIDs for fast and accurate matching.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, 
                 error_handler: Optional[ErrorHandler] = None):
        """
        Initialize the ExactMatcher.
        
        Args:
            logger: Optional logger instance for logging operations
            error_handler: Optional error handler for recovery strategies
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = error_handler or ErrorHandler(self.logger)
        self.uid_generator = UIDGenerator()
        self._block_mapping_cache = {}
        self._village_mapping_cache = {}
        self._processing_stats = {
            'total_processed': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'errors_encountered': 0
        }
    
    def match(self, entities: pd.DataFrame, lgd_data: pd.DataFrame) -> List[MappingResult]:
        """
        Perform exact matching on entities using LGD data.
        
        Args:
            entities: DataFrame containing entity records to match
            lgd_data: DataFrame containing LGD reference data
            
        Returns:
            List of MappingResult objects with exact matches
            
        Raises:
            MappingProcessError: If the matching process fails critically
        """
        self.logger.info("Starting exact matching process")
        
        try:
            # Reset processing stats
            self._processing_stats = {
                'total_processed': 0,
                'successful_matches': 0,
                'failed_matches': 0,
                'errors_encountered': 0
            }
            
            # Validate input data
            self._validate_input_data(entities, lgd_data)
            
            # Create mapping lookups with error handling
            block_mapping = self._create_block_mapping_safe(lgd_data)
            village_mapping = self._create_village_mapping_safe(lgd_data)
            
            # Perform matching with comprehensive error handling
            results = self._process_entities_safe(entities, block_mapping, village_mapping)
            
            # Log final statistics
            self._log_processing_statistics(results)
            
            return results
            
        except Exception as e:
            context = create_error_context(
                operation="exact_matching",
                entity_count=len(entities) if not entities.empty else 0,
                lgd_count=len(lgd_data) if not lgd_data.empty else 0,
                processed_count=self._processing_stats['total_processed']
            )
            
            log_error_details(self.logger, e, context)
            
            raise MappingProcessError(
                f"Exact matching process failed: {str(e)}",
                strategy="exact",
                entity_count=len(entities) if not entities.empty else 0,
                processed_count=self._processing_stats['total_processed'],
                original_error=e
            )
    
    def _validate_input_data(self, entities: pd.DataFrame, lgd_data: pd.DataFrame) -> None:
        """
        Validate input data for matching process.
        
        Args:
            entities: Entity DataFrame to validate
            lgd_data: LGD data DataFrame to validate
            
        Raises:
            ValidationError: If input data is invalid
        """
        if entities.empty:
            raise ValidationError(
                "Empty entities DataFrame provided to exact matcher",
                field_name="entities",
                invalid_value="empty_dataframe"
            )
        
        if lgd_data.empty:
            raise ValidationError(
                "Empty LGD data DataFrame provided to exact matcher",
                field_name="lgd_data",
                invalid_value="empty_dataframe"
            )
        
        # Check required columns
        required_entity_columns = ['district', 'block', 'village']
        missing_entity_cols = set(required_entity_columns) - set(entities.columns)
        if missing_entity_cols:
            raise ValidationError(
                f"Missing required columns in entities data: {missing_entity_cols}",
                field_name="entities_columns",
                invalid_value=list(entities.columns),
                validation_rules=[f"Must contain: {required_entity_columns}"]
            )
        
        required_lgd_columns = ['district_code', 'district', 'block_code', 'block', 'village_code', 'village']
        missing_lgd_cols = set(required_lgd_columns) - set(lgd_data.columns)
        if missing_lgd_cols:
            raise ValidationError(
                f"Missing required columns in LGD data: {missing_lgd_cols}",
                field_name="lgd_data_columns",
                invalid_value=list(lgd_data.columns),
                validation_rules=[f"Must contain: {required_lgd_columns}"]
            )
    
    def _create_block_mapping_safe(self, lgd_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Create block mapping with error handling.
        
        Args:
            lgd_data: LGD data DataFrame
            
        Returns:
            Block mapping dictionary
        """
        try:
            self.logger.info("Creating block mapping lookup")
            return self._create_block_mapping(lgd_data)
        except Exception as e:
            context = create_error_context(
                operation="create_block_mapping",
                lgd_record_count=len(lgd_data)
            )
            
            # Try to recover with partial mapping
            try:
                result = self.error_handler.handle_error(
                    e, context, recovery_strategy='partial_processing'
                )
                if result:
                    return result
            except Exception:
                pass
            
            # If recovery fails, raise the original error
            raise MatchingError(
                f"Failed to create block mapping: {str(e)}",
                matcher_type="exact",
                original_error=e
            )
    
    def _create_village_mapping_safe(self, lgd_data: pd.DataFrame) -> Dict[str, LGDRecord]:
        """
        Create village mapping with error handling.
        
        Args:
            lgd_data: LGD data DataFrame
            
        Returns:
            Village mapping dictionary
        """
        try:
            self.logger.info("Creating village mapping lookup")
            return self._create_village_mapping(lgd_data)
        except Exception as e:
            context = create_error_context(
                operation="create_village_mapping",
                lgd_record_count=len(lgd_data)
            )
            
            # Try to recover with partial mapping
            try:
                result = self.error_handler.handle_error(
                    e, context, recovery_strategy='partial_processing'
                )
                if result:
                    return result
            except Exception:
                pass
            
            # If recovery fails, raise the original error
            raise MatchingError(
                f"Failed to create village mapping: {str(e)}",
                matcher_type="exact",
                original_error=e
            )
    
    def _process_entities_safe(self, entities: pd.DataFrame, block_mapping: Dict[str, Dict], 
                              village_mapping: Dict[str, LGDRecord]) -> List[MappingResult]:
        """
        Process entities with comprehensive error handling.
        
        Args:
            entities: Entity DataFrame to process
            block_mapping: Block mapping lookup
            village_mapping: Village mapping lookup
            
        Returns:
            List of MappingResult objects
        """
        results = []
        total_entities = len(entities)
        
        self.logger.info(f"Processing {total_entities} entities for exact matching")
        
        with tqdm(total=total_entities, desc="Exact matching") as pbar:
            for idx, row in entities.iterrows():
                self._processing_stats['total_processed'] += 1
                
                try:
                    result = self._process_single_entity(idx, row, block_mapping, village_mapping)
                    results.append(result)
                    
                    if result.is_matched():
                        self._processing_stats['successful_matches'] += 1
                    else:
                        self._processing_stats['failed_matches'] += 1
                        
                except Exception as e:
                    self._processing_stats['errors_encountered'] += 1
                    
                    # Create error context
                    context = create_error_context(
                        operation="process_entity",
                        row_index=idx,
                        entity_data=row.to_dict(),
                        record_info=f"entity at row {idx}"
                    )
                    
                    # Try to handle the error
                    try:
                        result = self.error_handler.handle_error(
                            e, context, recovery_strategy='skip_record'
                        )
                        
                        # Create unmatched result for skipped record
                        entity = self._create_entity_safe(row)
                        results.append(MappingResult(
                            entity=entity,
                            lgd_match=None,
                            match_type='unmatched'
                        ))
                        self._processing_stats['failed_matches'] += 1
                        
                    except Exception as recovery_error:
                        self.logger.error(f"Failed to recover from error at row {idx}: {recovery_error}")
                        
                        # Create minimal result to continue processing
                        entity = EntityRecord(
                            district=str(row.get('district', 'ERROR')),
                            block=str(row.get('block', 'ERROR')),
                            village=str(row.get('village', 'ERROR')),
                            district_code=row.get('district_code')
                        )
                        results.append(MappingResult(
                            entity=entity,
                            lgd_match=None,
                            match_type='unmatched'
                        ))
                        self._processing_stats['failed_matches'] += 1
                
                pbar.update(1)
        
        return results
    
    def _process_single_entity(self, idx: int, row: pd.Series, block_mapping: Dict[str, Dict], 
                              village_mapping: Dict[str, LGDRecord]) -> MappingResult:
        """
        Process a single entity with error handling.
        
        Args:
            idx: Row index
            row: Entity data row
            block_mapping: Block mapping lookup
            village_mapping: Village mapping lookup
            
        Returns:
            MappingResult for the entity
        """
        try:
            # Create EntityRecord
            entity = EntityRecord(
                district=row['district'],
                block=row['block'],
                village=row['village'],
                district_code=row.get('district_code')
            )
            
            # Validate entity
            if not entity.is_valid():
                validation_errors = entity.get_validation_errors()
                self.logger.debug(f"Invalid entity at row {idx}: {validation_errors}")
                return MappingResult(
                    entity=entity,
                    lgd_match=None,
                    match_type='unmatched'
                )
            
            # Attempt exact matching
            lgd_match = self._find_exact_match(entity, block_mapping, village_mapping)
            
            if lgd_match:
                return MappingResult(
                    entity=entity,
                    lgd_match=lgd_match,
                    match_type='exact',
                    match_score=1.0
                )
            else:
                return MappingResult(
                    entity=entity,
                    lgd_match=None,
                    match_type='unmatched'
                )
                
        except Exception as e:
            # Create entity with available data
            entity = self._create_entity_safe(row)
            
            raise MatchingError(
                f"Error processing entity at row {idx}: {str(e)}",
                entity_data=row.to_dict(),
                matcher_type="exact",
                original_error=e
            )
    
    def _create_entity_safe(self, row: pd.Series) -> EntityRecord:
        """
        Safely create EntityRecord from row data.
        
        Args:
            row: Data row
            
        Returns:
            EntityRecord object
        """
        try:
            return EntityRecord(
                district=str(row.get('district', '')),
                block=str(row.get('block', '')),
                village=str(row.get('village', '')),
                district_code=row.get('district_code')
            )
        except Exception as e:
            self.logger.warning(f"Error creating EntityRecord, using defaults: {e}")
            return EntityRecord(
                district='ERROR',
                block='ERROR',
                village='ERROR',
                district_code=None
            )
    
    def _log_processing_statistics(self, results: List[MappingResult]) -> None:
        """
        Log comprehensive processing statistics.
        
        Args:
            results: List of mapping results
        """
        total_entities = self._processing_stats['total_processed']
        matched_count = sum(1 for r in results if r.is_matched())
        match_rate = (matched_count / total_entities) * 100 if total_entities > 0 else 0
        error_rate = (self._processing_stats['errors_encountered'] / total_entities) * 100 if total_entities > 0 else 0
        
        self.logger.info("Exact matching completed:")
        self.logger.info(f"  Total processed: {total_entities}")
        self.logger.info(f"  Successful matches: {matched_count} ({match_rate:.2f}%)")
        self.logger.info(f"  Failed matches: {self._processing_stats['failed_matches']}")
        self.logger.info(f"  Errors encountered: {self._processing_stats['errors_encountered']} ({error_rate:.2f}%)")
        
        # Log error summary from error handler
        error_summary = self.error_handler.get_error_summary()
        if error_summary['total_errors'] > 0:
            self.logger.info(f"Error summary: {error_summary}")
    
    def match_by_uid(self, entities_df: pd.DataFrame, lgd_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform UID-based matching and return DataFrame with match results.
        
        Args:
            entities_df: DataFrame containing entities with UIDs
            lgd_df: DataFrame containing LGD data with UIDs
            
        Returns:
            DataFrame with matching results
        """
        self.logger.info("Performing UID-based matching")
        
        # Ensure UID columns exist
        if 'uid' not in entities_df.columns:
            self.logger.warning("No UID column found in entities DataFrame")
            return entities_df.copy()
        
        if 'uid' not in lgd_df.columns:
            self.logger.warning("No UID column found in LGD DataFrame")
            return entities_df.copy()
        
        # Create lookup dictionary from LGD data
        lgd_lookup = {}
        for _, row in lgd_df.iterrows():
            uid = row.get('uid')
            if not is_null_or_empty(uid):
                lgd_lookup[uid] = row.to_dict()
        
        self.logger.info(f"Created lookup with {len(lgd_lookup)} LGD UIDs")
        
        # Perform matching
        result_df = entities_df.copy()
        result_df['match_found'] = False
        result_df['match_type'] = 'unmatched'
        result_df['lgd_district_code'] = None
        result_df['lgd_block_code'] = None
        result_df['lgd_village_code'] = None
        
        matched_count = 0
        
        for idx, row in result_df.iterrows():
            entity_uid = row.get('uid')
            
            if is_null_or_empty(entity_uid):
                continue
            
            if entity_uid in lgd_lookup:
                lgd_match = lgd_lookup[entity_uid]
                result_df.at[idx, 'match_found'] = True
                result_df.at[idx, 'match_type'] = 'exact'
                result_df.at[idx, 'lgd_district_code'] = lgd_match.get('district_code')
                result_df.at[idx, 'lgd_block_code'] = lgd_match.get('block_code')
                result_df.at[idx, 'lgd_village_code'] = lgd_match.get('village_code')
                matched_count += 1
        
        match_rate = (matched_count / len(result_df)) * 100 if len(result_df) > 0 else 0
        self.logger.info(f"UID matching completed: {matched_count}/{len(result_df)} "
                        f"matches found ({match_rate:.2f}%)")
        
        return result_df
    
    def _create_block_mapping(self, lgd_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Create block-level mapping lookup using district_code + block name.
        
        Args:
            lgd_data: DataFrame containing LGD reference data
            
        Returns:
            Dictionary mapping block UIDs to LGD block information
        """
        try:
            block_mapping = {}
            error_count = 0
            
            # Group by district_code and block to get unique block combinations
            try:
                block_groups = lgd_data.groupby(['district_code', 'block']).first().reset_index()
            except Exception as e:
                raise MatchingError(
                    f"Error grouping LGD data for block mapping: {str(e)}",
                    matcher_type="exact",
                    original_error=e
                )
            
            for idx, row in block_groups.iterrows():
                try:
                    district_code = row['district_code']
                    block_name = row['block']
                    
                    # Validate required fields
                    if pd.isna(district_code) or pd.isna(block_name) or str(block_name).strip() == '':
                        error_count += 1
                        self.logger.debug(f"Skipping block mapping for invalid data at index {idx}: "
                                        f"district_code={district_code}, block={block_name}")
                        continue
                    
                    # Generate block UID
                    block_uid = self.uid_generator.generate_district_block_uid(district_code, block_name)
                    
                    if block_uid:
                        block_mapping[block_uid] = {
                            'district_code': district_code,
                            'district': row['district'],
                            'block_code': row['block_code'],
                            'block': block_name
                        }
                    else:
                        error_count += 1
                        self.logger.debug(f"Failed to generate UID for block: {district_code}_{block_name}")
                        
                except Exception as e:
                    error_count += 1
                    self.logger.warning(f"Error processing block mapping at index {idx}: {e}")
                    continue
            
            if error_count > 0:
                self.logger.warning(f"Encountered {error_count} errors while creating block mapping")
            
            self.logger.info(f"Created block mapping with {len(block_mapping)} entries")
            self._block_mapping_cache = block_mapping
            return block_mapping
            
        except Exception as e:
            self.logger.error(f"Critical error creating block mapping: {e}")
            raise
    
    def _create_village_mapping(self, lgd_data: pd.DataFrame) -> Dict[str, LGDRecord]:
        """
        Create village-level mapping lookup using complete UIDs.
        
        Args:
            lgd_data: DataFrame containing LGD reference data
            
        Returns:
            Dictionary mapping village UIDs to LGDRecord objects
        """
        try:
            village_mapping = {}
            error_count = 0
            validation_error_count = 0
            
            for idx, row in lgd_data.iterrows():
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
                        validation_error_count += 1
                        validation_errors = lgd_record.get_validation_errors()
                        self.logger.debug(f"Invalid LGD record at index {idx}: {validation_errors}")
                        continue
                    
                    # Generate village UID using district_code + block_name + village_name
                    try:
                        village_uid = self.uid_generator.generate_full_village_uid(
                            lgd_record.district_code,
                            lgd_record.block,
                            lgd_record.village
                        )
                        
                        if village_uid:
                            # Check for duplicate UIDs
                            if village_uid in village_mapping:
                                self.logger.debug(f"Duplicate village UID found: {village_uid}")
                                # Keep the first occurrence
                                continue
                            
                            village_mapping[village_uid] = lgd_record
                        else:
                            error_count += 1
                            self.logger.debug(f"Failed to generate UID for village: "
                                            f"{lgd_record.district_code}_{lgd_record.block}_{lgd_record.village}")
                            
                    except Exception as uid_error:
                        error_count += 1
                        self.logger.debug(f"Error generating village UID at index {idx}: {uid_error}")
                        continue
                    
                except Exception as e:
                    error_count += 1
                    context = create_error_context(
                        operation="create_lgd_record",
                        row_index=idx,
                        row_data=row.to_dict()
                    )
                    log_error_details(self.logger, e, context)
                    continue
            
            # Log statistics
            total_processed = len(lgd_data)
            success_count = len(village_mapping)
            
            if error_count > 0:
                self.logger.warning(f"Encountered {error_count} errors while creating village mapping")
            if validation_error_count > 0:
                self.logger.warning(f"Skipped {validation_error_count} invalid LGD records")
            
            self.logger.info(f"Created village mapping with {success_count} entries "
                           f"from {total_processed} LGD records")
            
            # Check if too many records failed
            if success_count < total_processed * 0.5:  # Less than 50% success rate
                self.logger.warning(f"Low success rate in village mapping creation: "
                                  f"{success_count}/{total_processed} ({success_count/total_processed*100:.1f}%)")
            
            self._village_mapping_cache = village_mapping
            return village_mapping
            
        except Exception as e:
            self.logger.error(f"Critical error creating village mapping: {e}")
            raise
    
    def _find_exact_match(self, entity: EntityRecord, block_mapping: Dict[str, Dict], 
                         village_mapping: Dict[str, LGDRecord]) -> Optional[LGDRecord]:
        """
        Find exact match for an entity using block and village mappings.
        
        Args:
            entity: EntityRecord to match
            block_mapping: Block-level mapping lookup
            village_mapping: Village-level mapping lookup
            
        Returns:
            LGDRecord if exact match found, None otherwise
        """
        # If entity has district_code, try direct village matching
        if entity.district_code is not None:
            village_uid = self.uid_generator.generate_full_village_uid(
                entity.district_code,
                entity.block,
                entity.village
            )
            
            if village_uid and village_uid in village_mapping:
                return village_mapping[village_uid]
        
        # Try block-level matching to get district_code, then village matching
        if entity.district_code is None:
            # We need to find the district_code first through block matching
            # This is a limitation - we need district_code for exact matching
            return None
        
        return None
    
    def get_strategy_name(self) -> str:
        """
        Get the name of this matching strategy.
        
        Returns:
            Strategy name string
        """
        return "exact"
    
    def get_matching_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the matching process.
        
        Returns:
            Dictionary with matching statistics
        """
        stats = {
            'block_mappings_created': len(self._block_mapping_cache),
            'village_mappings_created': len(self._village_mapping_cache),
            'processing_stats': self._processing_stats.copy()
        }
        
        # Add UID generator statistics if available
        try:
            uid_stats = self.uid_generator.get_uid_statistics()
            stats['uid_statistics'] = uid_stats
        except Exception as e:
            self.logger.debug(f"Error getting UID statistics: {e}")
            stats['uid_statistics'] = {}
        
        # Add error handler statistics
        try:
            error_stats = self.error_handler.get_error_summary()
            stats['error_statistics'] = error_stats
        except Exception as e:
            self.logger.debug(f"Error getting error statistics: {e}")
            stats['error_statistics'] = {}
        
        return stats
    
    def clear_cache(self):
        """Clear internal mapping caches and reset statistics."""
        self._block_mapping_cache.clear()
        self._village_mapping_cache.clear()
        
        # Reset processing statistics
        self._processing_stats = {
            'total_processed': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'errors_encountered': 0
        }
        
        # Clear UID generator registry
        try:
            self.uid_generator.clear_registry()
        except Exception as e:
            self.logger.debug(f"Error clearing UID generator registry: {e}")
        
        # Reset error handler counters
        try:
            self.error_handler.reset_error_counts()
        except Exception as e:
            self.logger.debug(f"Error resetting error handler: {e}")
        
        self.logger.debug("Cleared ExactMatcher caches and statistics")
    
    def get_error_recovery_report(self) -> Dict[str, Any]:
        """
        Get detailed error recovery report.
        
        Returns:
            Dictionary with error recovery information
        """
        try:
            error_summary = self.error_handler.get_error_summary()
            
            return {
                'total_errors': error_summary.get('total_errors', 0),
                'error_breakdown': error_summary.get('error_counts', {}),
                'most_common_error': error_summary.get('most_common_error'),
                'processing_stats': self._processing_stats.copy(),
                'recovery_success_rate': self._calculate_recovery_success_rate()
            }
        except Exception as e:
            self.logger.warning(f"Error generating recovery report: {e}")
            return {
                'error': f"Failed to generate recovery report: {str(e)}",
                'processing_stats': self._processing_stats.copy()
            }
    
    def _calculate_recovery_success_rate(self) -> float:
        """
        Calculate the success rate of error recovery.
        
        Returns:
            Recovery success rate as percentage
        """
        total_processed = self._processing_stats['total_processed']
        errors_encountered = self._processing_stats['errors_encountered']
        
        if errors_encountered == 0:
            return 100.0
        
        if total_processed == 0:
            return 0.0
        
        # Recovery is considered successful if we processed the record despite errors
        recovery_success_rate = ((total_processed - errors_encountered) / total_processed) * 100
        return max(0.0, min(100.0, recovery_success_rate))