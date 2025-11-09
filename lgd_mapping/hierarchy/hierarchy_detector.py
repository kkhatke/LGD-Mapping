"""
Hierarchy detection for LGD mapping application.

This module provides functionality to automatically detect hierarchical structure
in input data by scanning for administrative level columns and validating
the detected hierarchy.
"""

import logging
import pandas as pd
from typing import List, Optional, Tuple

from lgd_mapping.hierarchy.hierarchy_config import HierarchyLevel, HierarchyConfiguration
from lgd_mapping.exceptions import (
    ValidationError, 
    HierarchyDetectionError,
    create_hierarchy_detection_error
)


class HierarchyDetector:
    """
    Detects and validates hierarchical structure in input data.
    
    This class scans entity and LGD DataFrames to identify which administrative
    levels are present (state, district, block, GP, village) and builds a
    HierarchyConfiguration that describes the detected structure.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the hierarchy detector.
        
        Args:
            logger: Optional logger instance for logging detection results
        """
        self.logger = logger or logging.getLogger(__name__)
        self.standard_levels = self._define_standard_levels()
    
    def detect_hierarchy(
        self, 
        entities_df: pd.DataFrame,
        lgd_df: pd.DataFrame
    ) -> HierarchyConfiguration:
        """
        Detect hierarchical levels present in data.
        
        This method scans both the entities and LGD DataFrames to identify which
        administrative levels are present. It checks for both name columns and
        code columns at each level.
        
        Args:
            entities_df: DataFrame containing entity records to be mapped
            lgd_df: DataFrame containing LGD reference codes
            
        Returns:
            HierarchyConfiguration with detected levels and configuration
            
        Raises:
            ValidationError: If the detected hierarchy is invalid or inconsistent
        """
        self.logger.info("Starting hierarchy detection...")
        
        detected_levels = []
        level_details = {}
        
        # Check each standard level
        for level in self.standard_levels:
            # Check if columns exist in both DataFrames
            entity_has_name, entity_has_code = self._detect_columns(entities_df, level)
            lgd_has_name, lgd_has_code = self._detect_columns(lgd_df, level)
            
            # Level is considered present if name column exists in entities
            # and at least name column exists in LGD data
            if entity_has_name and lgd_has_name:
                detected_levels.append(level.name)
                level_details[level.name] = {
                    'entity_has_name': entity_has_name,
                    'entity_has_code': entity_has_code,
                    'lgd_has_name': lgd_has_name,
                    'lgd_has_code': lgd_has_code
                }
                
                self.logger.info(
                    f"Detected level '{level.name}': "
                    f"entity_name={entity_has_name}, entity_code={entity_has_code}, "
                    f"lgd_name={lgd_has_name}, lgd_code={lgd_has_code}"
                )
            else:
                self.logger.debug(
                    f"Level '{level.name}' not detected: "
                    f"entity_name={entity_has_name}, lgd_name={lgd_has_name}"
                )
        
        # Validate the detected hierarchy
        is_valid, issues = self._validate_hierarchy_consistency(detected_levels)
        
        if not is_valid:
            # Identify missing required levels
            missing_required = [
                level.name for level in self.standard_levels 
                if level.is_required and level.name not in detected_levels
            ]
            
            self.logger.error(
                f"Hierarchy detection failed: {len(issues)} validation issue(s) found",
                extra={
                    'detected_levels': detected_levels,
                    'missing_required_levels': missing_required,
                    'validation_issues': issues
                }
            )
            
            raise create_hierarchy_detection_error(
                detected_levels=detected_levels,
                missing_required=missing_required,
                inconsistencies=issues
            )
        
        # Build hierarchy configuration
        hierarchy_config = HierarchyConfiguration(
            levels=self.standard_levels,
            detected_levels=detected_levels,
            enable_code_mapping={level: True for level in detected_levels},
            fuzzy_thresholds={
                level.name: level.fuzzy_threshold 
                for level in self.standard_levels 
                if level.fuzzy_threshold is not None
            }
        )
        
        # Log summary with hierarchical context
        self.logger.info(
            f"Hierarchy detection complete: {len(detected_levels)} level(s) detected",
            extra={
                'detected_levels': detected_levels,
                'hierarchy_depth': hierarchy_config.get_hierarchy_depth(),
                'level_details': level_details
            }
        )
        self.logger.info(f"  Detected levels: {', '.join(detected_levels)}")
        self.logger.info(f"  Hierarchy depth: {hierarchy_config.get_hierarchy_depth()}")
        
        # Log details for each detected level
        for level_name in detected_levels:
            details = level_details[level_name]
            code_status = "present" if details['entity_has_code'] else "needs mapping"
            self.logger.info(
                f"  {level_name}: code {code_status}",
                extra={
                    'hierarchy_level': level_name,
                    'code_present': details['entity_has_code'],
                    'requires_mapping': not details['entity_has_code']
                }
            )
        
        return hierarchy_config
    
    def _define_standard_levels(self) -> List[HierarchyLevel]:
        """
        Define standard Indian administrative hierarchy.
        
        Returns:
            List of HierarchyLevel objects representing the standard hierarchy
            from state (top) to village (bottom).
            
        Note:
            Only district is marked as required. At least one lower level
            (block, gp, or village) must be present for valid hierarchy.
        """
        return [
            HierarchyLevel(
                name='state',
                code_column='state_code',
                name_column='state',  # In entities it's 'state', in LGD it's 'state_name'
                is_required=False,
                parent_level=None,
                fuzzy_threshold=85
            ),
            HierarchyLevel(
                name='district',
                code_column='district_code',
                name_column='district',
                is_required=True,
                parent_level='state',
                fuzzy_threshold=85
            ),
            HierarchyLevel(
                name='subdistrict',
                code_column='subdistrict_code',
                name_column='subdistrict',
                is_required=False,
                parent_level='district',
                fuzzy_threshold=90
            ),
            HierarchyLevel(
                name='block',
                code_column='block_code',
                name_column='block',
                is_required=False,  # Changed to False - at least one lower level must be present
                parent_level='district',
                fuzzy_threshold=90
            ),
            HierarchyLevel(
                name='gp',
                code_column='gp_code',
                name_column='gp',
                is_required=False,
                parent_level='block',
                fuzzy_threshold=90
            ),
            HierarchyLevel(
                name='village',
                code_column='village_code',
                name_column='village',
                is_required=False,  # Changed to False - at least one lower level must be present
                parent_level='gp',
                fuzzy_threshold=95
            )
        ]
    
    def _detect_columns(
        self, 
        df: pd.DataFrame, 
        level: HierarchyLevel
    ) -> Tuple[bool, bool]:
        """
        Check if level's columns are present in DataFrame.
        
        Args:
            df: DataFrame to check for columns
            level: HierarchyLevel to check
            
        Returns:
            Tuple of (has_name_column, has_code_column)
        """
        has_name = level.name_column in df.columns
        has_code = level.code_column in df.columns
        
        # Special case: LGD file uses 'state_name' instead of 'state'
        if level.name == 'state' and not has_name and 'state_name' in df.columns:
            has_name = True
        
        return has_name, has_code
    
    def _validate_hierarchy_consistency(
        self,
        detected_levels: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that detected hierarchy is consistent.
        
        This method checks:
        1. All required levels are present (only district is required)
        2. No gaps in the hierarchy (if a child is present, parent should be too)
        
        Args:
            detected_levels: List of detected level names
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for required levels based on standard definitions (only district)
        for level in self.standard_levels:
            if level.is_required and level.name not in detected_levels:
                issues.append(f"Required level '{level.name}' not detected in data")
        
        # Check for hierarchy gaps (if child is present, parent should be too)
        # Build a map of level to its position in the standard hierarchy
        level_positions = {level.name: idx for idx, level in enumerate(self.standard_levels)}
        
        for level_name in detected_levels:
            level = next((l for l in self.standard_levels if l.name == level_name), None)
            if level and level.parent_level:
                # Check if parent is present
                parent_level = level.parent_level
                
                # Special case: village can have either gp or block as parent
                if level_name == 'village' and parent_level == 'gp':
                    if 'gp' not in detected_levels and 'block' not in detected_levels:
                        issues.append(
                            f"Level '{level_name}' detected but neither parent 'gp' "
                            f"nor 'block' is present"
                        )
                # Special case: gp and subdistrict are optional intermediate levels
                elif level_name in ['gp', 'subdistrict']:
                    # These can be skipped, so don't enforce parent requirement strictly
                    pass
                else:
                    # For other levels, check if parent is present
                    if parent_level not in detected_levels:
                        # Check if any ancestor is present
                        parent_obj = next(
                            (l for l in self.standard_levels if l.name == parent_level), 
                            None
                        )
                        if parent_obj and parent_obj.parent_level:
                            # If grandparent exists, that's acceptable
                            if parent_obj.parent_level not in detected_levels:
                                self.logger.warning(
                                    f"Level '{level_name}' detected but parent '{parent_level}' "
                                    f"is not present. This may affect matching accuracy."
                                )
        
        # Check for logical inconsistencies
        # If both block and subdistrict are present, log a warning
        if 'block' in detected_levels and 'subdistrict' in detected_levels:
            self.logger.warning(
                "Both 'block' and 'subdistrict' detected. These are typically "
                "alternative administrative levels. Proceeding with both."
            )
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            self.logger.error(f"Hierarchy validation failed with {len(issues)} issue(s):")
            for issue in issues:
                self.logger.error(f"  - {issue}")
        else:
            self.logger.info("Hierarchy validation passed")
        
        return is_valid, issues
