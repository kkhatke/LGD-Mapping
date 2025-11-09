"""
Hierarchical UID generation for LGD mapping application.

This module provides functionality for generating unique identifiers based on
variable-length administrative hierarchies, supporting flexible combinations
from state down to village level.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from ..utils.uid_generator import UIDGenerator
from ..utils.data_utils import safe_int_conversion, safe_string_conversion, is_null_or_empty
from .hierarchy_config import HierarchyConfiguration, HierarchyLevel


class HierarchicalUIDGenerator:
    """
    Generates UIDs based on hierarchical configuration with variable depth.
    
    Supports generating UIDs from any combination of hierarchical levels
    (state, district, block, GP, village) while maintaining backward
    compatibility with existing 3-level UIDs.
    """
    
    def __init__(
        self,
        hierarchy_config: HierarchyConfiguration,
        separator: str = "_",
        enable_batch_processing: bool = True,
        batch_size: int = 10000
    ):
        """
        Initialize the hierarchical UID generator.
        
        Args:
            hierarchy_config: Configuration defining the hierarchy structure
            separator: Character used to separate UID components (default: "_")
            enable_batch_processing: Enable batch processing for large datasets
            batch_size: Number of records to process in each batch
        """
        self.hierarchy_config = hierarchy_config
        self.separator = separator
        self.uid_generator = UIDGenerator(separator)
        self.enable_batch_processing = enable_batch_processing
        self.batch_size = batch_size
        self._generation_stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0,
            'missing_components': 0
        }
        self._component_cache = {}  # Cache for component extraction
    
    def generate_uid(
        self,
        record: Union[pd.Series, Dict],
        uid_type: str = 'full'
    ) -> Optional[str]:
        """
        Generate UID from record based on available hierarchy.
        
        Args:
            record: Record containing hierarchical data (Series or Dict)
            uid_type: 'full' for all detected levels, 'partial' for available levels
            
        Returns:
            Generated UID string or None if insufficient data
        """
        self._generation_stats['total_attempted'] += 1
        
        # Collect UID components based on hierarchy
        components = self._collect_uid_components(record)
        
        if not components:
            self._generation_stats['missing_components'] += 1
            return None
        
        # Generate UID by joining components
        try:
            uid = self.separator.join(components)
            self._generation_stats['successful'] += 1
            return uid
        except Exception:
            self._generation_stats['failed'] += 1
            return None
    
    def generate_uids_for_dataframe(
        self,
        df: pd.DataFrame,
        uid_column: str = 'uid'
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Add UIDs to all records in DataFrame with optimized batch processing.
        
        Args:
            df: DataFrame to add UIDs to
            uid_column: Name of column to store generated UIDs
            
        Returns:
            Tuple of (updated DataFrame, generation statistics)
        """
        df_result = df.copy()
        
        # Reset statistics
        self._generation_stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0,
            'missing_components': 0
        }
        
        # Use batch processing for large datasets
        if self.enable_batch_processing and len(df_result) > self.batch_size:
            uids = self._generate_uids_batch(df_result)
        else:
            # Generate UID for each row (original approach)
            uids = []
            for idx, row in df_result.iterrows():
                uid = self.generate_uid(row, uid_type='full')
                uids.append(uid)
        
        # Add UIDs to DataFrame
        df_result[uid_column] = uids
        
        # Calculate statistics
        stats = {
            'total_records': len(df_result),
            'uids_generated': self._generation_stats['successful'],
            'generation_failed': self._generation_stats['failed'],
            'missing_components': self._generation_stats['missing_components'],
            'success_rate': (
                self._generation_stats['successful'] / len(df_result) * 100
                if len(df_result) > 0 else 0.0
            ),
            'hierarchy_depth': self.hierarchy_config.get_hierarchy_depth(),
            'detected_levels': self.hierarchy_config.detected_levels.copy(),
            'batch_processing_used': self.enable_batch_processing and len(df_result) > self.batch_size
        }
        
        return df_result, stats
    
    def _collect_uid_components(
        self,
        record: Union[pd.Series, Dict]
    ) -> List[str]:
        """
        Collect UID components from record based on hierarchy configuration.
        
        Extracts code values for each detected hierarchical level in order,
        cleaning and validating each component.
        
        Args:
            record: Record containing hierarchical data
            
        Returns:
            List of cleaned component strings in hierarchical order
        """
        components = []
        
        # Get detected levels in hierarchical order
        detected_levels = self.hierarchy_config.get_detected_levels_in_order()
        
        for level in detected_levels:
            # Get the code column value for this level
            code_column = level.code_column
            
            # Handle both Series and Dict access
            try:
                if isinstance(record, pd.Series):
                    value = record.get(code_column)
                else:
                    value = record.get(code_column)
            except (KeyError, AttributeError):
                # Column not found, skip this level
                continue
            
            # Skip if value is null or empty
            if is_null_or_empty(value):
                continue
            
            # Convert and clean the component
            cleaned = self._clean_component(value, level)
            
            if cleaned:
                components.append(cleaned)
        
        return components
    
    def _clean_component(
        self,
        value: any,
        level: HierarchyLevel
    ) -> Optional[str]:
        """
        Clean and validate a UID component value.
        
        Args:
            value: Raw component value
            level: HierarchyLevel defining this component
            
        Returns:
            Cleaned component string or None if invalid
        """
        if is_null_or_empty(value):
            return None
        
        # For code columns, convert to integer then string
        if 'code' in level.code_column.lower():
            converted = safe_int_conversion(value)
            if converted is None:
                return None
            return str(converted)
        
        # For name columns, use the existing cleaner
        return self.uid_generator._clean_component(value)
    
    def parse_hierarchical_uid(
        self,
        uid: str
    ) -> Optional[Dict[str, str]]:
        """
        Parse UID into hierarchical components.
        
        Decomposes a UID string into its constituent parts based on the
        hierarchy configuration, mapping each component to its level.
        
        Args:
            uid: UID string to parse
            
        Returns:
            Dictionary mapping level names to their values, or None if invalid
        """
        if is_null_or_empty(uid):
            return None
        
        uid_str = safe_string_conversion(uid)
        
        # Split UID into components
        components = uid_str.split(self.separator)
        
        # Get detected levels in order
        detected_levels = self.hierarchy_config.get_detected_levels_in_order()
        
        # Check if component count matches detected levels
        if len(components) != len(detected_levels):
            return None
        
        # Map components to levels
        parsed = {}
        for i, level in enumerate(detected_levels):
            parsed[level.name] = components[i]
            parsed[level.code_column] = components[i]
        
        # Add metadata
        parsed['hierarchy_depth'] = len(components)
        parsed['hierarchy_levels'] = [level.name for level in detected_levels]
        
        return parsed
    
    def validate_uid_hierarchy(
        self,
        uid: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate UID structure against hierarchy configuration.
        
        Checks that the UID has the correct number of components and that
        each component is valid for its hierarchical level.
        
        Args:
            uid: UID string to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if UID is empty
        if is_null_or_empty(uid):
            issues.append("UID is empty or null")
            return False, issues
        
        uid_str = safe_string_conversion(uid)
        
        # Check for separator
        if self.separator not in uid_str:
            issues.append(f"UID must contain separator '{self.separator}'")
            return False, issues
        
        # Split into components
        components = uid_str.split(self.separator)
        
        # Get expected number of components
        detected_levels = self.hierarchy_config.get_detected_levels_in_order()
        expected_count = len(detected_levels)
        
        # Check component count
        if len(components) != expected_count:
            issues.append(
                f"UID has {len(components)} components but expected {expected_count} "
                f"based on hierarchy: {[level.name for level in detected_levels]}"
            )
            return False, issues
        
        # Validate each component
        for i, (component, level) in enumerate(zip(components, detected_levels)):
            # Check if component is empty
            if not component.strip():
                issues.append(f"Component {i} ({level.name}) is empty")
                continue
            
            # For code columns, validate it's numeric
            if 'code' in level.code_column.lower():
                try:
                    int(component)
                except ValueError:
                    issues.append(
                        f"Component {i} ({level.name}) should be numeric but got: {component}"
                    )
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_generation_statistics(self) -> Dict[str, any]:
        """
        Get statistics about UID generation.
        
        Returns:
            Dictionary containing generation statistics
        """
        return self._generation_stats.copy()
    
    def is_backward_compatible_uid(self, uid: str) -> bool:
        """
        Check if UID is in the legacy 3-level format.
        
        Args:
            uid: UID string to check
            
        Returns:
            True if UID has 3 components (district_code, block_code, village)
        """
        if is_null_or_empty(uid):
            return False
        
        uid_str = safe_string_conversion(uid)
        components = uid_str.split(self.separator)
        
        return len(components) == 3

    def _generate_uids_batch(self, df: pd.DataFrame) -> List[Optional[str]]:
        """
        Generate UIDs for DataFrame using optimized batch processing.
        
        This method processes records in batches and uses vectorized operations
        where possible for better performance with large datasets.
        
        Args:
            df: DataFrame to generate UIDs for
            
        Returns:
            List of generated UIDs
        """
        uids = []
        total_records = len(df)
        
        # Get detected levels in order
        detected_levels = self.hierarchy_config.get_detected_levels_in_order()
        code_columns = [level.code_column for level in detected_levels]
        
        # Process in batches
        for batch_start in range(0, total_records, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_records)
            batch_df = df.iloc[batch_start:batch_end]
            
            # Extract all code columns for this batch
            batch_components = []
            
            for _, row in batch_df.iterrows():
                self._generation_stats['total_attempted'] += 1
                
                # Collect components for this row
                components = []
                has_all_components = True
                
                for level in detected_levels:
                    code_column = level.code_column
                    
                    try:
                        value = row.get(code_column)
                    except (KeyError, AttributeError):
                        has_all_components = False
                        break
                    
                    if is_null_or_empty(value):
                        has_all_components = False
                        break
                    
                    # Convert and clean the component
                    cleaned = self._clean_component(value, level)
                    
                    if cleaned:
                        components.append(cleaned)
                    else:
                        has_all_components = False
                        break
                
                # Generate UID from components
                if components and has_all_components:
                    try:
                        uid = self.separator.join(components)
                        batch_components.append(uid)
                        self._generation_stats['successful'] += 1
                    except Exception:
                        batch_components.append(None)
                        self._generation_stats['failed'] += 1
                else:
                    batch_components.append(None)
                    self._generation_stats['missing_components'] += 1
            
            uids.extend(batch_components)
        
        return uids
    
    def clear_cache(self):
        """
        Clear component cache to free memory.
        
        This should be called when switching to a new dataset or when
        memory needs to be freed.
        """
        self._component_cache.clear()
    
    def get_performance_statistics(self) -> Dict[str, any]:
        """
        Get performance statistics for UID generation.
        
        Returns:
            Dictionary with performance statistics
        """
        return {
            'batch_processing_enabled': self.enable_batch_processing,
            'batch_size': self.batch_size,
            'cache_size': len(self._component_cache),
            'generation_stats': self._generation_stats.copy()
        }
