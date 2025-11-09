"""
Exact matching strategy for LGD mapping.

This module provides the ExactMatcher class that performs direct UID-based matching
for both block-level and village-level entities with support for hierarchical UIDs.
"""

import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple, Set, Any, TYPE_CHECKING
from tqdm import tqdm

from ..models import EntityRecord, LGDRecord, MappingResult
from ..exceptions import MatchingError, ValidationError, MappingProcessError
from ..utils.uid_generator import UIDGenerator
from ..utils.data_utils import is_null_or_empty
from ..utils.error_handler import (
    ErrorHandler, with_error_handling, create_error_context, log_error_details
)

if TYPE_CHECKING:
    from ..hierarchy.hierarchy_config import HierarchyConfiguration
    from ..hierarchy.hierarchical_uid_generator import HierarchicalUIDGenerator


class ExactMatcher:
    """
    Performs exact matching using unique identifiers (UIDs) for administrative entities.
    
    This class implements both block-level and village-level exact matching strategies
    using generated UIDs for fast and accurate matching.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, 
                 error_handler: Optional[ErrorHandler] = None,
                 hierarchy_config: Optional['HierarchyConfiguration'] = None,
                 hierarchical_uid_generator: Optional['HierarchicalUIDGenerator'] = None,
                 enable_progressive_filtering: bool = True,
                 enable_batch_processing: bool = True,
                 batch_size: int = 5000):
        """
        Initialize the ExactMatcher with performance optimizations.
        
        Args:
            logger: Optional logger instance for logging operations
            error_handler: Optional error handler for recovery strategies
            hierarchy_config: Optional hierarchy configuration for hierarchical matching
            hierarchical_uid_generator: Optional hierarchical UID generator
            enable_progressive_filtering: Enable progressive filtering by parent codes
            enable_batch_processing: Enable batch processing by parent codes
            batch_size: Number of records to process in each batch
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = error_handler or ErrorHandler(self.logger)
        self.uid_generator = UIDGenerator()
        self.hierarchy_config = hierarchy_config
        self.hierarchical_uid_generator = hierarchical_uid_generator
        self.enable_progressive_filtering = enable_progressive_filtering
        self.enable_batch_processing = enable_batch_processing
        self.batch_size = batch_size
        self._block_mapping_cache = {}
        self._village_mapping_cache = {}
        self._hierarchical_mapping_cache = {}
        self._filtered_lgd_cache = {}  # Cache for progressively filtered LGD data
        self._processing_stats = {
            'total_processed': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'errors_encountered': 0,
            'hierarchical_matches': 0,
            'legacy_matches': 0,
            'hierarchy_validation_failures': 0,
            'progressive_filtering_used': 0,
            'batch_processing_used': 0
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
                'errors_encountered': 0,
                'hierarchical_matches': 0,
                'legacy_matches': 0,
                'hierarchy_validation_failures': 0,
                'progressive_filtering_used': 0,
                'batch_processing_used': 0
            }
            
            # Validate input data
            self._validate_input_data(entities, lgd_data)
            
            # Determine if we should use hierarchical matching
            use_hierarchical = self._should_use_hierarchical_matching(entities, lgd_data)
            
            if use_hierarchical:
                self.logger.info("Using hierarchical UID matching")
                # Create hierarchical mapping lookup
                hierarchical_mapping = self._create_hierarchical_mapping_safe(lgd_data)
                
                # Perform hierarchical matching with batch processing if enabled
                if self.enable_batch_processing and len(entities) >= self.batch_size:
                    self.logger.info(f"Using batch processing (batch_size={self.batch_size})")
                    results = self._process_entities_in_batches(entities, hierarchical_mapping)
                else:
                    results = self._process_entities_hierarchical(entities, hierarchical_mapping)
            else:
                self.logger.info("Using legacy 3-level UID matching")
                # Create mapping lookups with error handling (legacy approach)
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
    
    def _should_use_hierarchical_matching(self, entities: pd.DataFrame, lgd_data: pd.DataFrame) -> bool:
        """
        Determine if hierarchical matching should be used.
        
        Hierarchical matching is used when:
        1. Hierarchy configuration is available
        2. Hierarchical UID generator is available
        3. UIDs are present in both dataframes
        
        Args:
            entities: Entity DataFrame
            lgd_data: LGD data DataFrame
            
        Returns:
            True if hierarchical matching should be used, False for legacy matching
        """
        # Check if hierarchical components are available
        if self.hierarchy_config is None or self.hierarchical_uid_generator is None:
            self.logger.debug("Hierarchical components not available, using legacy matching")
            return False
        
        # Check if UIDs are present in both dataframes
        if 'uid' not in entities.columns or 'uid' not in lgd_data.columns:
            self.logger.debug("UID columns not found, using legacy matching")
            return False
        
        # Check if hierarchy depth is greater than 3 (beyond legacy support)
        hierarchy_depth = self.hierarchy_config.get_hierarchy_depth()
        if hierarchy_depth > 3:
            self.logger.info(f"Detected {hierarchy_depth}-level hierarchy, using hierarchical matching")
            return True
        
        # For 3-level hierarchy, check if UIDs are populated
        uid_count = entities['uid'].notna().sum()
        if uid_count > 0:
            self.logger.info("UIDs present in data, using hierarchical matching")
            return True
        
        self.logger.debug("Using legacy matching for 3-level hierarchy without UIDs")
        return False
    
    def _create_hierarchical_mapping_safe(self, lgd_data: pd.DataFrame) -> Dict[str, 'LGDRecord']:
        """
        Create hierarchical mapping lookup with error handling.
        
        Args:
            lgd_data: LGD data DataFrame with UIDs
            
        Returns:
            Dictionary mapping UIDs to LGDRecord objects
        """
        try:
            self.logger.info("Creating hierarchical mapping lookup")
            return self._create_hierarchical_mapping(lgd_data)
        except Exception as e:
            context = create_error_context(
                operation="create_hierarchical_mapping",
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
                f"Failed to create hierarchical mapping: {str(e)}",
                matcher_type="exact",
                original_error=e
            )
    
    def _create_hierarchical_mapping(self, lgd_data: pd.DataFrame) -> Dict[str, 'LGDRecord']:
        """
        Create hierarchical mapping lookup using UIDs with multi-level indexing.
        
        Creates both a UID-based mapping and parent-code-based indexes for
        progressive filtering during matching.
        
        Args:
            lgd_data: LGD data DataFrame with UIDs
            
        Returns:
            Dictionary mapping UIDs to LGDRecord objects
        """
        hierarchical_mapping = {}
        error_count = 0
        validation_error_count = 0
        
        # Create multi-level indexes for progressive filtering
        if self.enable_progressive_filtering and self.hierarchy_config:
            self._create_multi_level_indexes(lgd_data)
        
        for idx, row in lgd_data.iterrows():
            try:
                # Get UID from row
                uid = row.get('uid')
                
                if is_null_or_empty(uid):
                    error_count += 1
                    self.logger.debug(f"Missing UID at index {idx}")
                    continue
                
                # Create LGDRecord with hierarchical fields
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
                
                # Validate record
                if not lgd_record.is_valid():
                    validation_error_count += 1
                    validation_errors = lgd_record.get_validation_errors()
                    self.logger.debug(f"Invalid LGD record at index {idx}: {validation_errors}")
                    continue
                
                # Check for duplicate UIDs
                if uid in hierarchical_mapping:
                    self.logger.debug(f"Duplicate UID found: {uid}")
                    continue
                
                hierarchical_mapping[uid] = lgd_record
                
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
        success_count = len(hierarchical_mapping)
        
        if error_count > 0:
            self.logger.warning(f"Encountered {error_count} errors while creating hierarchical mapping")
        if validation_error_count > 0:
            self.logger.warning(f"Skipped {validation_error_count} invalid LGD records")
        
        self.logger.info(f"Created hierarchical mapping with {success_count} entries "
                       f"from {total_processed} LGD records")
        
        # Check if too many records failed
        if success_count < total_processed * 0.5:
            self.logger.warning(f"Low success rate in hierarchical mapping creation: "
                              f"{success_count}/{total_processed} ({success_count/total_processed*100:.1f}%)")
        
        self._hierarchical_mapping_cache = hierarchical_mapping
        return hierarchical_mapping
    
    def _process_entities_hierarchical(
        self, 
        entities: pd.DataFrame, 
        hierarchical_mapping: Dict[str, 'LGDRecord']
    ) -> List[MappingResult]:
        """
        Process entities using hierarchical UID matching.
        
        Args:
            entities: Entity DataFrame with UIDs
            hierarchical_mapping: Dictionary mapping UIDs to LGDRecord objects
            
        Returns:
            List of MappingResult objects
        """
        results = []
        total_entities = len(entities)
        
        self.logger.info(f"Processing {total_entities} entities for hierarchical exact matching")
        
        with tqdm(total=total_entities, desc="Hierarchical exact matching") as pbar:
            for idx, row in entities.iterrows():
                self._processing_stats['total_processed'] += 1
                
                try:
                    result = self._process_single_entity_hierarchical(idx, row, hierarchical_mapping)
                    results.append(result)
                    
                    if result.is_matched():
                        self._processing_stats['successful_matches'] += 1
                        self._processing_stats['hierarchical_matches'] += 1
                    else:
                        self._processing_stats['failed_matches'] += 1
                        
                except Exception as e:
                    self._processing_stats['errors_encountered'] += 1
                    
                    # Create error context
                    context = create_error_context(
                        operation="process_entity_hierarchical",
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
                        entity = self._create_entity_from_row_hierarchical(row)
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
    
    def _process_single_entity_hierarchical(
        self, 
        idx: int, 
        row: pd.Series, 
        hierarchical_mapping: Dict[str, 'LGDRecord']
    ) -> MappingResult:
        """
        Process a single entity using hierarchical UID matching.
        
        Args:
            idx: Row index
            row: Entity data row
            hierarchical_mapping: Dictionary mapping UIDs to LGDRecord objects
            
        Returns:
            MappingResult for the entity
        """
        try:
            # Create EntityRecord with hierarchical fields
            entity = self._create_entity_from_row_hierarchical(row)
            
            # Validate entity
            if not entity.is_valid():
                validation_errors = entity.get_validation_errors()
                self.logger.debug(f"Invalid entity at row {idx}: {validation_errors}")
                return MappingResult(
                    entity=entity,
                    lgd_match=None,
                    match_type='unmatched'
                )
            
            # Get UID from row
            entity_uid = row.get('uid')
            
            if is_null_or_empty(entity_uid):
                self.logger.debug(f"Missing UID for entity at row {idx}")
                return MappingResult(
                    entity=entity,
                    lgd_match=None,
                    match_type='unmatched'
                )
            
            # Validate UID structure if hierarchical UID generator is available
            if self.hierarchical_uid_generator:
                is_valid, issues = self.hierarchical_uid_generator.validate_uid_hierarchy(entity_uid)
                if not is_valid:
                    self.logger.debug(f"Invalid UID structure at row {idx}: {issues}")
                    self._processing_stats['hierarchy_validation_failures'] += 1
                    # Continue with matching anyway - UID might still work
            
            # Attempt hierarchical exact matching
            lgd_match = hierarchical_mapping.get(entity_uid)
            
            if lgd_match:
                # Create hierarchy match details
                hierarchy_match_details = self._create_hierarchy_match_details(entity, lgd_match)
                
                return MappingResult(
                    entity=entity,
                    lgd_match=lgd_match,
                    match_type='exact',
                    match_score=1.0,
                    hierarchy_match_details=hierarchy_match_details
                )
            else:
                return MappingResult(
                    entity=entity,
                    lgd_match=None,
                    match_type='unmatched'
                )
                
        except Exception as e:
            # Create entity with available data
            entity = self._create_entity_from_row_hierarchical(row)
            
            raise MatchingError(
                f"Error processing entity at row {idx}: {str(e)}",
                entity_data=row.to_dict(),
                matcher_type="exact",
                original_error=e
            )
    
    def _create_entity_from_row_hierarchical(self, row: pd.Series) -> EntityRecord:
        """
        Create EntityRecord from row data with hierarchical fields.
        
        Args:
            row: Data row
            
        Returns:
            EntityRecord object with hierarchical fields populated
        """
        try:
            return EntityRecord(
                district=str(row.get('district', '')),
                block=str(row.get('block', '')),
                village=str(row.get('village', '')),
                state=row.get('state'),
                gp=row.get('gp'),
                subdistrict=row.get('subdistrict'),
                state_code=row.get('state_code'),
                district_code=row.get('district_code'),
                block_code=row.get('block_code'),
                gp_code=row.get('gp_code'),
                village_code=row.get('village_code')
            )
        except Exception as e:
            self.logger.warning(f"Error creating EntityRecord with hierarchical fields, using defaults: {e}")
            return EntityRecord(
                district='ERROR',
                block='ERROR',
                village='ERROR',
                district_code=None
            )
    
    def _create_hierarchy_match_details(
        self, 
        entity: EntityRecord, 
        lgd_match: 'LGDRecord'
    ) -> Dict[str, str]:
        """
        Create hierarchy match details for a successful match.
        
        Compares each hierarchical level between entity and LGD match to
        determine how each level was matched (exact, fuzzy, or unmatched).
        
        Args:
            entity: Matched entity record
            lgd_match: Matched LGD record
            
        Returns:
            Dictionary mapping level names to match types
        """
        match_details = {}
        
        # Check each hierarchical level
        if entity.state and lgd_match.state:
            match_details['state'] = 'exact'
        
        if entity.district and lgd_match.district:
            match_details['district'] = 'exact'
        
        if entity.subdistrict and lgd_match.subdistrict:
            match_details['subdistrict'] = 'exact'
        
        if entity.block and lgd_match.block:
            match_details['block'] = 'exact'
        
        if entity.gp and lgd_match.gp:
            match_details['gp'] = 'exact'
        
        if entity.village and lgd_match.village:
            match_details['village'] = 'exact'
        
        return match_details
    
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
                        self._processing_stats['legacy_matches'] += 1
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
            # Create EntityRecord with all hierarchical fields
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
                village_code=row.get('village_code'),
                subdistrict_code=row.get('subdistrict_code')
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
                state=row.get('state'),
                gp=row.get('gp'),
                subdistrict=row.get('subdistrict'),
                state_code=row.get('state_code'),
                district_code=row.get('district_code'),
                block_code=row.get('block_code'),
                gp_code=row.get('gp_code'),
                village_code=row.get('village_code'),
                subdistrict_code=row.get('subdistrict_code')
            )
        except Exception as e:
            self.logger.warning(f"Error creating EntityRecord, using defaults: {e}")
            return EntityRecord(
                district='ERROR',
                block='ERROR',
                village='ERROR',
                state=None,
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
        
        # Log with hierarchical context
        hierarchy_depth = self.hierarchy_config.get_hierarchy_depth() if self.hierarchy_config else 3
        detected_levels = self.hierarchy_config.detected_levels if self.hierarchy_config else ['district', 'block', 'village']
        
        self.logger.info(
            f"Exact matching completed: {matched_count}/{total_entities} matched ({match_rate:.2f}%)",
            extra={
                'total_processed': total_entities,
                'successful_matches': matched_count,
                'match_rate': match_rate,
                'failed_matches': self._processing_stats['failed_matches'],
                'errors_encountered': self._processing_stats['errors_encountered'],
                'error_rate': error_rate,
                'hierarchy_depth': hierarchy_depth,
                'detected_levels': detected_levels,
                'hierarchical_matches': self._processing_stats['hierarchical_matches'],
                'legacy_matches': self._processing_stats['legacy_matches']
            }
        )
        
        self.logger.info(f"  Total processed: {total_entities}")
        self.logger.info(f"  Successful matches: {matched_count} ({match_rate:.2f}%)")
        self.logger.info(f"  Failed matches: {self._processing_stats['failed_matches']}")
        self.logger.info(f"  Errors encountered: {self._processing_stats['errors_encountered']} ({error_rate:.2f}%)")
        
        # Log hierarchical matching statistics if applicable
        if self._processing_stats['hierarchical_matches'] > 0:
            hierarchical_rate = (
                self._processing_stats['hierarchical_matches'] / total_entities * 100
                if total_entities > 0 else 0
            )
            self.logger.info(
                f"  Hierarchical matches: {self._processing_stats['hierarchical_matches']} "
                f"({hierarchical_rate:.2f}%)",
                extra={
                    'hierarchical_matches': self._processing_stats['hierarchical_matches'],
                    'hierarchical_match_rate': hierarchical_rate,
                    'hierarchy_depth': hierarchy_depth
                }
            )
        
        if self._processing_stats['legacy_matches'] > 0:
            legacy_rate = (
                self._processing_stats['legacy_matches'] / total_entities * 100
                if total_entities > 0 else 0
            )
            self.logger.info(f"  Legacy matches: {self._processing_stats['legacy_matches']} "
                           f"({legacy_rate:.2f}%)")
        
        if self._processing_stats['hierarchy_validation_failures'] > 0:
            self.logger.warning(
                f"  Hierarchy validation failures: {self._processing_stats['hierarchy_validation_failures']}",
                extra={
                    'hierarchy_validation_failures': self._processing_stats['hierarchy_validation_failures'],
                    'validation_failure_rate': (self._processing_stats['hierarchy_validation_failures'] / total_entities * 100) if total_entities > 0 else 0
                }
            )
        
        # Log performance optimization statistics
        if self._processing_stats['progressive_filtering_used'] > 0:
            self.logger.info(f"  Progressive filtering used: "
                           f"{self._processing_stats['progressive_filtering_used']} times")
        
        if self._processing_stats['batch_processing_used'] > 0:
            self.logger.info(f"  Batch processing used: "
                           f"{self._processing_stats['batch_processing_used']} batches")
        
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
            'hierarchical_mappings_created': len(self._hierarchical_mapping_cache),
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
        self._hierarchical_mapping_cache.clear()
        self._filtered_lgd_cache.clear()
        
        # Reset processing statistics
        self._processing_stats = {
            'total_processed': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'errors_encountered': 0,
            'hierarchical_matches': 0,
            'legacy_matches': 0,
            'hierarchy_validation_failures': 0,
            'progressive_filtering_used': 0,
            'batch_processing_used': 0
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
   
    def _create_multi_level_indexes(self, lgd_data: pd.DataFrame):
        """
        Create multi-level indexes for progressive filtering.
        
        Creates indexes at each hierarchical level to enable efficient
        filtering by parent codes during matching.
        
        Args:
            lgd_data: LGD data DataFrame
        """
        if not self.hierarchy_config:
            return
        
        self.logger.info("Creating multi-level indexes for progressive filtering")
        
        # Create indexes for each detected level
        detected_levels = self.hierarchy_config.detected_levels
        
        for i, level in enumerate(detected_levels):
            if i == 0:
                # Top level - no parent filtering needed
                continue
            
            # Get parent level
            parent_level = detected_levels[i - 1]
            parent_code_col = self.hierarchy_config.get_level(parent_level).code_column
            
            # Create index grouped by parent code
            if parent_code_col in lgd_data.columns:
                grouped = lgd_data.groupby(parent_code_col).groups
                index_key = f"index_{level}_{parent_level}"
                self._filtered_lgd_cache[index_key] = grouped
                
                self.logger.debug(
                    f"Created index for {level} grouped by {parent_level}: "
                    f"{len(grouped)} parent groups"
                )
        
        self.logger.info(f"Created {len(self._filtered_lgd_cache)} multi-level indexes")
    
    def _get_filtered_lgd_by_parent(
        self, 
        lgd_data: pd.DataFrame, 
        level: str, 
        parent_code: any
    ) -> pd.DataFrame:
        """
        Get LGD data filtered by parent code using progressive filtering.
        
        Args:
            lgd_data: Full LGD DataFrame
            level: Current level name
            parent_code: Parent code to filter by
            
        Returns:
            Filtered DataFrame containing only records with matching parent code
        """
        if not self.enable_progressive_filtering or not self.hierarchy_config:
            return lgd_data
        
        # Find parent level
        detected_levels = self.hierarchy_config.detected_levels
        level_index = detected_levels.index(level) if level in detected_levels else -1
        
        if level_index <= 0:
            return lgd_data
        
        parent_level = detected_levels[level_index - 1]
        index_key = f"index_{level}_{parent_level}"
        
        # Use cached index if available
        if index_key in self._filtered_lgd_cache:
            grouped = self._filtered_lgd_cache[index_key]
            if parent_code in grouped:
                indices = grouped[parent_code]
                self._processing_stats['progressive_filtering_used'] += 1
                return lgd_data.loc[indices]
        
        # Fallback to direct filtering
        parent_code_col = self.hierarchy_config.get_level(parent_level).code_column
        if parent_code_col in lgd_data.columns:
            filtered = lgd_data[lgd_data[parent_code_col] == parent_code]
            self._processing_stats['progressive_filtering_used'] += 1
            return filtered
        
        return lgd_data
    
    def _process_entities_in_batches(
        self,
        entities: pd.DataFrame,
        hierarchical_mapping: Dict[str, 'LGDRecord']
    ) -> List[MappingResult]:
        """
        Process entities in batches grouped by parent codes for better performance.
        
        Groups entities by their parent codes and processes each group together,
        reducing redundant lookups and improving cache efficiency.
        
        Args:
            entities: Entity DataFrame
            hierarchical_mapping: Hierarchical mapping dictionary
            
        Returns:
            List of MappingResult objects
        """
        results = []
        
        if not self.hierarchy_config or len(entities) < self.batch_size:
            # Fall back to regular processing
            return self._process_entities_hierarchical(entities, hierarchical_mapping)
        
        self.logger.info(f"Processing {len(entities)} entities in batches by parent codes")
        
        # Get the second level (e.g., district) for grouping
        detected_levels = self.hierarchy_config.detected_levels
        if len(detected_levels) < 2:
            return self._process_entities_hierarchical(entities, hierarchical_mapping)
        
        grouping_level = detected_levels[0]  # Use top level for grouping
        grouping_code_col = self.hierarchy_config.get_level(grouping_level).code_column
        
        if grouping_code_col not in entities.columns:
            return self._process_entities_hierarchical(entities, hierarchical_mapping)
        
        # Group entities by parent code
        grouped = entities.groupby(grouping_code_col)
        total_groups = len(grouped)
        
        self.logger.info(f"Processing {total_groups} batches grouped by {grouping_level}")
        
        from tqdm import tqdm
        with tqdm(total=len(entities), desc="Batch exact matching") as pbar:
            for parent_code, group_df in grouped:
                self._processing_stats['batch_processing_used'] += 1
                
                # Process this batch
                for idx, row in group_df.iterrows():
                    self._processing_stats['total_processed'] += 1
                    
                    try:
                        result = self._process_single_entity_hierarchical(
                            idx, row, hierarchical_mapping
                        )
                        results.append(result)
                        
                        if result.is_matched():
                            self._processing_stats['successful_matches'] += 1
                            self._processing_stats['hierarchical_matches'] += 1
                        else:
                            self._processing_stats['failed_matches'] += 1
                    
                    except Exception as e:
                        self._processing_stats['errors_encountered'] += 1
                        
                        # Create error context
                        context = create_error_context(
                            operation="process_entity_hierarchical_batch",
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
                            entity = self._create_entity_from_row_hierarchical(row)
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
