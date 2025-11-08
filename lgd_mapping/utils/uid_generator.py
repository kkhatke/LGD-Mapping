"""
UID generation and management utilities for LGD mapping application.

This module provides functionality for generating unique identifiers for
district-block and village combinations, handling null values and special
characters, and validating UIDs.
"""

import re
from typing import Optional, List, Set, Dict, Tuple
import pandas as pd
from .data_utils import safe_string_conversion, safe_int_conversion, is_null_or_empty


class UIDGenerator:
    """
    Handles generation and management of unique identifiers for administrative entities.
    """
    
    def __init__(self, separator: str = "_"):
        """
        Initialize the UID generator.
        
        Args:
            separator: Character used to separate UID components
        """
        self.separator = separator
        self._generated_uids = set()
        self._uid_registry = {}  # Maps UIDs to their components
    
    def _clean_component(self, component: str) -> str:
        """
        Clean a UID component by removing/replacing problematic characters.
        
        Args:
            component: Component string to clean
            
        Returns:
            Cleaned component string
        """
        if is_null_or_empty(component):
            return ""
        
        # Convert to string and clean
        cleaned = safe_string_conversion(component)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Replace problematic characters with underscores
        # Keep alphanumeric, spaces, hyphens, and dots
        cleaned = re.sub(r'[^\w\s\-\.]', '_', cleaned)
        
        # Replace multiple spaces/underscores with single underscore
        cleaned = re.sub(r'[\s_]+', '_', cleaned)
        
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        return cleaned
    
    def _validate_components(self, *components) -> Tuple[bool, str]:
        """
        Validate UID components.
        
        Args:
            *components: Variable number of components to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not components:
            return False, "No components provided"
        
        for i, component in enumerate(components):
            if component is None:
                return False, f"Component {i} is None"
            
            # Convert to string for validation
            str_component = str(component)
            
            if not str_component.strip():
                return False, f"Component {i} is empty or whitespace"
        
        return True, ""
    
    def generate_district_block_uid(self, district_code: int, block_name: str) -> Optional[str]:
        """
        Generate UID for district-block combination.
        
        Args:
            district_code: Numeric district code
            block_name: Name of the block
            
        Returns:
            Generated UID or None if invalid inputs
        """
        # Validate inputs
        is_valid, error = self._validate_components(district_code, block_name)
        if not is_valid:
            return None
        
        # Convert and clean components
        converted_code = safe_int_conversion(district_code)
        if converted_code is None:
            return None
        code_str = str(converted_code)
        
        cleaned_block = self._clean_component(block_name)
        if not cleaned_block:
            return None
        
        # Generate UID
        uid = f"{code_str}{self.separator}{cleaned_block}"
        
        # Register the UID
        self._generated_uids.add(uid)
        self._uid_registry[uid] = {
            'type': 'district_block',
            'district_code': district_code,
            'block_name': block_name
        }
        
        return uid
    
    def generate_village_uid(self, district_code: int, block_code: int, village_name: str) -> Optional[str]:
        """
        Generate UID for district-block-village combination.
        
        Args:
            district_code: Numeric district code
            block_code: Numeric block code
            village_name: Name of the village
            
        Returns:
            Generated UID or None if invalid inputs
        """
        # Validate inputs
        is_valid, error = self._validate_components(district_code, block_code, village_name)
        if not is_valid:
            return None
        
        # Convert and clean components
        converted_district = safe_int_conversion(district_code)
        converted_block = safe_int_conversion(block_code)
        
        if converted_district is None or converted_block is None:
            return None
            
        district_str = str(converted_district)
        block_str = str(converted_block)
        
        cleaned_village = self._clean_component(village_name)
        if not cleaned_village:
            return None
        
        # Generate UID
        uid = f"{district_str}{self.separator}{block_str}{self.separator}{cleaned_village}"
        
        # Register the UID
        self._generated_uids.add(uid)
        self._uid_registry[uid] = {
            'type': 'village',
            'district_code': district_code,
            'block_code': block_code,
            'village_name': village_name
        }
        
        return uid
    
    def generate_full_village_uid(self, district_code: int, block_name: str, village_name: str) -> Optional[str]:
        """
        Generate UID for district-block-village combination using block name instead of code.
        
        Args:
            district_code: Numeric district code
            block_name: Name of the block
            village_name: Name of the village
            
        Returns:
            Generated UID or None if invalid inputs
        """
        # Validate inputs
        is_valid, error = self._validate_components(district_code, block_name, village_name)
        if not is_valid:
            return None
        
        # Convert and clean components
        converted_district = safe_int_conversion(district_code)
        if converted_district is None:
            return None
        district_str = str(converted_district)
        
        cleaned_block = self._clean_component(block_name)
        cleaned_village = self._clean_component(village_name)
        
        if not cleaned_block or not cleaned_village:
            return None
        
        # Generate UID
        uid = f"{district_str}{self.separator}{cleaned_block}{self.separator}{cleaned_village}"
        
        # Register the UID
        self._generated_uids.add(uid)
        self._uid_registry[uid] = {
            'type': 'full_village',
            'district_code': district_code,
            'block_name': block_name,
            'village_name': village_name
        }
        
        return uid
    
    def validate_uid_format(self, uid: str) -> Tuple[bool, str]:
        """
        Validate UID format and structure.
        
        Args:
            uid: UID string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if is_null_or_empty(uid):
            return False, "UID is empty or null"
        
        uid_str = safe_string_conversion(uid)
        
        # Check for separator
        if self.separator not in uid_str:
            return False, f"UID must contain separator '{self.separator}'"
        
        # Split into components
        components = uid_str.split(self.separator)
        
        # Check minimum components (district + at least one other)
        if len(components) < 2:
            return False, "UID must have at least 2 components"
        
        # Check maximum components (district + block + village)
        if len(components) > 3:
            return False, "UID cannot have more than 3 components"
        
        # Validate first component is numeric (district code)
        try:
            int(components[0])
        except ValueError:
            return False, "First component must be numeric (district code)"
        
        # Check that all components are non-empty
        for i, component in enumerate(components):
            if not component.strip():
                return False, f"Component {i} is empty"
        
        return True, ""
    
    def parse_uid(self, uid: str) -> Optional[Dict[str, str]]:
        """
        Parse UID into its components.
        
        Args:
            uid: UID string to parse
            
        Returns:
            Dictionary with UID components or None if invalid
        """
        is_valid, error = self.validate_uid_format(uid)
        if not is_valid:
            return None
        
        components = uid.split(self.separator)
        
        if len(components) == 2:
            return {
                'district_code': components[0],
                'block_name': components[1],
                'type': 'district_block'
            }
        elif len(components) == 3:
            # Could be either village (district_code + block_code + village) 
            # or full_village (district_code + block_name + village)
            try:
                int(components[1])  # If second component is numeric, it's a block code
                return {
                    'district_code': components[0],
                    'block_code': components[1],
                    'village_name': components[2],
                    'type': 'village'
                }
            except ValueError:
                return {
                    'district_code': components[0],
                    'block_name': components[1],
                    'village_name': components[2],
                    'type': 'full_village'
                }
        
        return None
    
    def detect_duplicate_uids(self, uids: List[str]) -> Dict[str, List[int]]:
        """
        Detect duplicate UIDs in a list.
        
        Args:
            uids: List of UIDs to check for duplicates
            
        Returns:
            Dictionary mapping duplicate UIDs to list of their indices
        """
        uid_indices = {}
        duplicates = {}
        
        for i, uid in enumerate(uids):
            if is_null_or_empty(uid):
                continue
            
            uid_str = safe_string_conversion(uid)
            
            if uid_str in uid_indices:
                if uid_str not in duplicates:
                    duplicates[uid_str] = [uid_indices[uid_str]]
                duplicates[uid_str].append(i)
            else:
                uid_indices[uid_str] = i
        
        return duplicates
    
    def get_uid_statistics(self) -> Dict[str, int]:
        """
        Get statistics about generated UIDs.
        
        Returns:
            Dictionary with UID statistics
        """
        type_counts = {}
        for uid_info in self._uid_registry.values():
            uid_type = uid_info.get('type', 'unknown')
            type_counts[uid_type] = type_counts.get(uid_type, 0) + 1
        
        return {
            'total_generated': len(self._generated_uids),
            'registered_uids': len(self._uid_registry),
            'type_breakdown': type_counts
        }
    
    def clear_registry(self):
        """Clear the UID registry and generated UIDs set."""
        self._generated_uids.clear()
        self._uid_registry.clear()
    
    def add_uids_to_dataframe(self, df: pd.DataFrame, uid_type: str, 
                             uid_column: str = 'uid', **kwargs) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Add UIDs to a DataFrame based on specified columns.
        
        Args:
            df: DataFrame to add UIDs to
            uid_type: Type of UID to generate ('district_block', 'village', 'full_village')
            uid_column: Name of column to store UIDs
            **kwargs: Column names for UID components
            
        Returns:
            Tuple of (updated_dataframe, generation_report)
        """
        df_result = df.copy()
        generation_report = {
            'generated': [],
            'failed': [],
            'invalid_inputs': []
        }
        
        for idx, row in df_result.iterrows():
            try:
                if uid_type == 'district_block':
                    district_code = kwargs.get('district_code_col', 'district_code')
                    block_name = kwargs.get('block_name_col', 'block')
                    
                    uid = self.generate_district_block_uid(
                        row[district_code], 
                        row[block_name]
                    )
                
                elif uid_type == 'village':
                    district_code = kwargs.get('district_code_col', 'district_code')
                    block_code = kwargs.get('block_code_col', 'block_code')
                    village_name = kwargs.get('village_name_col', 'village')
                    
                    uid = self.generate_village_uid(
                        row[district_code],
                        row[block_code],
                        row[village_name]
                    )
                
                elif uid_type == 'full_village':
                    district_code = kwargs.get('district_code_col', 'district_code')
                    block_name = kwargs.get('block_name_col', 'block')
                    village_name = kwargs.get('village_name_col', 'village')
                    
                    uid = self.generate_full_village_uid(
                        row[district_code],
                        row[block_name],
                        row[village_name]
                    )
                
                else:
                    df_result.at[idx, uid_column] = None
                    generation_report['failed'].append(f"Row {idx}: Invalid UID type: {uid_type}")
                    continue
                
                if uid:
                    df_result.at[idx, uid_column] = uid
                    generation_report['generated'].append(uid)
                else:
                    df_result.at[idx, uid_column] = None
                    generation_report['failed'].append(f"Row {idx}: Failed to generate UID")
                    
            except KeyError as e:
                df_result.at[idx, uid_column] = None
                generation_report['invalid_inputs'].append(f"Row {idx}: Missing column {e}")
            except Exception as e:
                df_result.at[idx, uid_column] = None
                generation_report['failed'].append(f"Row {idx}: {str(e)}")
        
        return df_result, generation_report


def create_uid_from_components(*components, separator: str = "_") -> Optional[str]:
    """
    Create a UID from individual components.
    
    Args:
        *components: Variable number of components
        separator: Separator character
        
    Returns:
        Generated UID or None if invalid
    """
    generator = UIDGenerator(separator)
    
    if len(components) == 2:
        # Assume district_code, block_name
        try:
            district_code = int(components[0])
            return generator.generate_district_block_uid(district_code, components[1])
        except (ValueError, TypeError):
            return None
    
    elif len(components) == 3:
        # Assume district_code, block_code/name, village_name
        try:
            district_code = int(components[0])
            try:
                block_code = int(components[1])
                return generator.generate_village_uid(district_code, block_code, components[2])
            except (ValueError, TypeError):
                return generator.generate_full_village_uid(district_code, components[1], components[2])
        except (ValueError, TypeError):
            return None
    
    return None


def validate_uid_list(uids: List[str], separator: str = "_") -> Dict[str, List[str]]:
    """
    Validate a list of UIDs and categorize them.
    
    Args:
        uids: List of UIDs to validate
        separator: Separator character used in UIDs
        
    Returns:
        Dictionary categorizing UIDs as valid, invalid, or duplicates
    """
    generator = UIDGenerator(separator)
    
    validation_report = {
        'valid': [],
        'invalid': [],
        'duplicates': [],
        'empty': []
    }
    
    # Check for duplicates
    duplicates = generator.detect_duplicate_uids(uids)
    duplicate_uids = set(duplicates.keys())
    
    for uid in uids:
        if is_null_or_empty(uid):
            validation_report['empty'].append(uid)
            continue
        
        uid_str = safe_string_conversion(uid)
        
        if uid_str in duplicate_uids:
            validation_report['duplicates'].append(uid_str)
            continue
        
        is_valid, error = generator.validate_uid_format(uid_str)
        if is_valid:
            validation_report['valid'].append(uid_str)
        else:
            validation_report['invalid'].append(uid_str)
    
    # Remove duplicates from each category
    for category in validation_report:
        validation_report[category] = list(set(validation_report[category]))
    
    return validation_report