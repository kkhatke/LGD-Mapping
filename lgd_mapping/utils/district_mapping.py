"""
District code mapping utilities for LGD mapping application.

This module provides functionality for mapping district names to codes,
handling variations in district names, and validating district assignments.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
from .data_utils import safe_string_conversion, normalize_string, is_null_or_empty


class DistrictMapper:
    """
    Handles district name to code mapping with support for name variations
    and standardization.
    """
    
    def __init__(self, district_code_mapping: Optional[Dict[str, int]] = None):
        """
        Initialize the district mapper.
        
        Args:
            district_code_mapping: Optional dictionary mapping district names to codes
        """
        self.district_code_mapping = district_code_mapping or {}
        self._normalized_mapping = {}
        self._variation_patterns = {}
        self._build_normalized_mapping()
    
    def _build_normalized_mapping(self):
        """Build normalized mapping for efficient lookups."""
        for district_name, code in self.district_code_mapping.items():
            normalized_name = self._normalize_district_name(district_name)
            self._normalized_mapping[normalized_name] = code
            
            # Generate common variations
            variations = self._generate_name_variations(district_name)
            for variation in variations:
                normalized_variation = self._normalize_district_name(variation)
                if normalized_variation not in self._normalized_mapping:
                    self._normalized_mapping[normalized_variation] = code
    
    def _normalize_district_name(self, name: str) -> str:
        """
        Normalize district name for consistent matching.
        
        Args:
            name: District name to normalize
            
        Returns:
            Normalized district name
        """
        if is_null_or_empty(name):
            return ""
        
        # Convert to string and clean
        name = safe_string_conversion(name)
        
        # Remove common prefixes/suffixes and normalize
        name = re.sub(r'\b(district|dist|zilla)\b', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Convert to title case for consistency
        return name.title()
    
    def _generate_name_variations(self, name: str) -> List[str]:
        """
        Generate common variations of a district name.
        
        Args:
            name: Original district name
            
        Returns:
            List of name variations
        """
        variations = []
        base_name = name.strip()
        
        # Add variations with common suffixes/prefixes
        variations.extend([
            f"{base_name} District",
            f"{base_name} Dist",
            f"{base_name} Zilla",
            f"District {base_name}",
            f"Dist {base_name}"
        ])
        
        # Add variations with different spacing
        variations.append(base_name.replace(' ', ''))
        variations.append(base_name.replace(' ', '-'))
        
        # Add uppercase and lowercase variations
        variations.extend([
            base_name.upper(),
            base_name.lower()
        ])
        
        return list(set(variations))  # Remove duplicates
    
    def add_district_mapping(self, district_name: str, district_code: int) -> bool:
        """
        Add a new district name to code mapping.
        
        Args:
            district_name: Name of the district
            district_code: Numeric code for the district
            
        Returns:
            True if mapping was added successfully, False if it already exists
        """
        if is_null_or_empty(district_name) or district_code is None:
            return False
        
        normalized_name = self._normalize_district_name(district_name)
        
        # Check if mapping already exists
        if normalized_name in self._normalized_mapping:
            return False
        
        # Add to both mappings
        self.district_code_mapping[district_name] = district_code
        self._normalized_mapping[normalized_name] = district_code
        
        # Add variations
        variations = self._generate_name_variations(district_name)
        for variation in variations:
            normalized_variation = self._normalize_district_name(variation)
            if normalized_variation not in self._normalized_mapping:
                self._normalized_mapping[normalized_variation] = district_code
        
        return True
    
    def get_district_code(self, district_name: str) -> Optional[int]:
        """
        Get district code for a given district name.
        
        Args:
            district_name: Name of the district to look up
            
        Returns:
            District code if found, None otherwise
        """
        if is_null_or_empty(district_name):
            return None
        
        normalized_name = self._normalize_district_name(district_name)
        return self._normalized_mapping.get(normalized_name)
    
    def validate_district_assignment(self, district_name: str, district_code: int) -> Tuple[bool, str]:
        """
        Validate that a district name matches the expected code.
        
        Args:
            district_name: Name of the district
            district_code: Expected district code
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if is_null_or_empty(district_name):
            return False, "District name is empty or null"
        
        if district_code is None:
            return False, "District code is null"
        
        mapped_code = self.get_district_code(district_name)
        
        if mapped_code is None:
            return False, f"No mapping found for district: {district_name}"
        
        if mapped_code != district_code:
            return False, f"District code mismatch: expected {mapped_code}, got {district_code}"
        
        return True, ""
    
    def get_unmapped_districts(self, district_names: List[str]) -> List[str]:
        """
        Get list of district names that don't have mappings.
        
        Args:
            district_names: List of district names to check
            
        Returns:
            List of unmapped district names
        """
        unmapped = []
        
        for name in district_names:
            if not is_null_or_empty(name) and self.get_district_code(name) is None:
                unmapped.append(name)
        
        return list(set(unmapped))  # Remove duplicates
    
    def get_mapping_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the current district mappings.
        
        Returns:
            Dictionary with mapping statistics
        """
        return {
            'total_base_mappings': len(self.district_code_mapping),
            'total_normalized_mappings': len(self._normalized_mapping),
            'unique_codes': len(set(self.district_code_mapping.values()))
        }
    
    def assign_district_codes_to_dataframe(self, df: pd.DataFrame, 
                                         district_column: str = 'district',
                                         code_column: str = 'district_code') -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Assign district codes to a DataFrame based on district names.
        
        Args:
            df: DataFrame containing district names
            district_column: Name of the column containing district names
            code_column: Name of the column to store district codes
            
        Returns:
            Tuple of (updated_dataframe, assignment_report)
        """
        if district_column not in df.columns:
            raise ValueError(f"District column '{district_column}' not found in DataFrame")
        
        df_result = df.copy()
        assignment_report = {
            'mapped': [],
            'unmapped': [],
            'invalid': []
        }
        
        # Assign codes
        for idx, row in df_result.iterrows():
            district_name = row[district_column]
            
            if is_null_or_empty(district_name):
                assignment_report['invalid'].append(f"Row {idx}: Empty district name")
                df_result.at[idx, code_column] = None
                continue
            
            district_code = self.get_district_code(district_name)
            
            if district_code is not None:
                df_result.at[idx, code_column] = district_code
                assignment_report['mapped'].append(district_name)
            else:
                df_result.at[idx, code_column] = None
                assignment_report['unmapped'].append(district_name)
        
        # Remove duplicates from report lists
        assignment_report['mapped'] = list(set(assignment_report['mapped']))
        assignment_report['unmapped'] = list(set(assignment_report['unmapped']))
        
        return df_result, assignment_report


def create_default_district_mapping() -> Dict[str, int]:
    """
    Create a default district mapping for common Indian districts.
    This is a sample mapping - in practice, this should be loaded from
    a configuration file or database.
    
    Returns:
        Dictionary mapping district names to codes
    """
    return {
        # Sample mappings - these should be replaced with actual LGD codes
        "Agra": 101,
        "Aligarh": 102,
        "Allahabad": 103,
        "Prayagraj": 103,  # Alternative name for Allahabad
        "Ambedkar Nagar": 104,
        "Amethi": 105,
        "Amroha": 106,
        "Auraiya": 107,
        "Azamgarh": 108,
        "Baghpat": 109,
        "Bahraich": 110,
        "Ballia": 111,
        "Balrampur": 112,
        "Banda": 113,
        "Barabanki": 114,
        "Bareilly": 115,
        "Basti": 116,
        "Bhadohi": 117,
        "Bijnor": 118,
        "Budaun": 119,
        "Bulandshahr": 120
    }


def load_district_mapping_from_file(file_path: str) -> Dict[str, int]:
    """
    Load district mapping from a CSV file.
    
    Args:
        file_path: Path to CSV file with columns 'district_name' and 'district_code'
        
    Returns:
        Dictionary mapping district names to codes
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    try:
        df = pd.read_csv(file_path)
        
        required_columns = ['district_name', 'district_code']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean and validate data
        mapping = {}
        for _, row in df.iterrows():
            district_name = safe_string_conversion(row['district_name'])
            district_code = row['district_code']
            
            if not is_null_or_empty(district_name) and pd.notna(district_code):
                try:
                    mapping[district_name] = int(district_code)
                except (ValueError, TypeError):
                    continue  # Skip invalid codes
        
        return mapping
        
    except FileNotFoundError:
        raise FileNotFoundError(f"District mapping file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading district mapping file: {str(e)}")