"""
District code mapper utility for LGD mapping application.

This module provides functionality to automatically map district names to district codes
using the LGD reference data, supporting both exact and fuzzy matching strategies.
"""

import logging
import re
from typing import Dict, Optional, Tuple, Any, List
import pandas as pd
from rapidfuzz import fuzz

from lgd_mapping.exceptions import ValidationError, DataQualityError


class DistrictCodeMapper:
    """
    Maps district names to district codes using LGD reference data.
    
    Supports both exact matching (after normalization) and fuzzy matching
    to handle spelling variations and inconsistent naming conventions.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the district code mapper.
        
        Args:
            logger: Optional logger instance for logging mapping operations
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def map_district_codes(
        self, 
        entities_df: pd.DataFrame, 
        lgd_df: pd.DataFrame,
        fuzzy_threshold: int = 85
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Map district names to codes and return updated DataFrame with statistics.
        
        Args:
            entities_df: Entity DataFrame with district names
            lgd_df: LGD reference DataFrame with district codes
            fuzzy_threshold: Minimum similarity score for fuzzy matching (default: 85)
            
        Returns:
            Tuple of (updated entities DataFrame, mapping statistics dict)
            
        Raises:
            ValidationError: If required columns are missing or DataFrames are empty
            DataQualityError: If mapping success rate is 0%
        """
        self.logger.info("Starting district code mapping process")
        
        # Validate entities DataFrame
        self._validate_entities_dataframe(entities_df)
        
        # Validate LGD DataFrame
        self._validate_lgd_dataframe(lgd_df)
        
        # Extract unique district names from entities
        unique_districts = entities_df['district'].dropna().unique()
        total_unique = len(unique_districts)
        
        self.logger.info(f"Found {total_unique} unique district names to map")
        
        # Build district code lookup from LGD data
        district_lookup = self._build_district_lookup(lgd_df)
        
        # Track mapping results
        mapping_details = {}
        exact_matches = 0
        fuzzy_matches = 0
        unmapped_districts = []
        
        # Map each unique district name to a code
        for district_name in unique_districts:
            # Extract all alternative names (including parenthetical text)
            alternative_names = self._extract_alternative_names(district_name)
            
            district_code = None
            match_type = None
            
            # Try exact match with all alternative names
            for alt_name in alternative_names:
                district_code = self._find_district_code_exact(alt_name, district_lookup)
                if district_code:
                    match_type = 'exact'
                    self.logger.debug(f"Exact match: '{district_name}' (using '{alt_name}') -> code {district_code}")
                    break
            
            # If no exact match, try fuzzy match with all alternative names
            if not district_code:
                for alt_name in alternative_names:
                    district_code = self._find_district_code_fuzzy(
                        alt_name, 
                        district_lookup, 
                        fuzzy_threshold
                    )
                    if district_code:
                        match_type = 'fuzzy'
                        self.logger.debug(f"Fuzzy match: '{district_name}' (using '{alt_name}') -> code {district_code}")
                        break
            
            # Track results
            if district_code:
                if match_type == 'exact':
                    exact_matches += 1
                else:
                    fuzzy_matches += 1
                mapping_details[district_name] = district_code
            else:
                unmapped_districts.append(district_name)
                self.logger.warning(f"Could not map district name: '{district_name}'")
        
        # Update entities DataFrame with mapped district codes
        entities_df['district_code'] = entities_df['district'].map(mapping_details)
        
        # Calculate statistics
        successfully_mapped = len(mapping_details)
        failed_to_map = total_unique - successfully_mapped
        success_rate = (successfully_mapped / total_unique * 100) if total_unique > 0 else 0
        
        stats = {
            'total_unique_districts': total_unique,
            'successfully_mapped': successfully_mapped,
            'failed_to_map': failed_to_map,
            'success_rate': success_rate,
            'exact_matches': exact_matches,
            'fuzzy_matches': fuzzy_matches,
            'unmapped_districts': unmapped_districts,
            'mapping_details': mapping_details
        }
        
        self.logger.info(
            f"District code mapping completed: {successfully_mapped}/{total_unique} mapped "
            f"({success_rate:.1f}% success rate, {exact_matches} exact, {fuzzy_matches} fuzzy)"
        )
        
        if failed_to_map > 0:
            self.logger.warning(f"{failed_to_map} districts could not be mapped")
        
        # Check for 0% mapping success rate and raise error
        if success_rate == 0 and total_unique > 0:
            affected_records = len(entities_df)
            recommendations = [
                "Review district names in input data for spelling errors",
                "Verify district names match LGD reference data",
                "Check for data encoding issues (e.g., special characters)",
                "Update LGD reference data if using outdated version"
            ]
            error_msg = (
                f"Failed to map any district names to codes. "
                f"Affected records: {affected_records}. "
                f"Recommendations: {'; '.join(recommendations)}"
            )
            self.logger.error(error_msg)
            raise DataQualityError(
                message=error_msg,
                quality_issue="district_mapping_failure",
                affected_records=affected_records,
                severity="critical",
                recommendations=recommendations
            )
        
        # Check for low mapping success rate and log warning
        if success_rate < 50 and total_unique > 0:
            affected_records = failed_to_map
            recommendations = [
                f"Review {failed_to_map} unmapped district names: {', '.join(unmapped_districts[:5])}{'...' if len(unmapped_districts) > 5 else ''}",
                "Consider updating LGD reference data",
                "Check for regional naming variations",
                "Lower fuzzy matching threshold if appropriate"
            ]
            warning_msg = (
                f"Low district code mapping success rate: {success_rate:.1f}%. "
                f"Affected records: {affected_records}. "
                f"Recommendations: {'; '.join(recommendations)}"
            )
            self.logger.warning(warning_msg)
        
        return entities_df, stats
    
    def _normalize_district_name(self, name: str) -> str:
        """
        Normalize district name for comparison.
        
        Removes parenthetical text, trims whitespace, converts to lowercase.
        
        Args:
            name: Original district name
            
        Returns:
            Normalized district name
        """
        if pd.isna(name) or not isinstance(name, str):
            return ""
        
        # Remove text in parentheses (e.g., "Keonjhar (Kendujhar)" -> "Keonjhar")
        normalized = re.sub(r'\s*\([^)]*\)', '', name)
        
        # Trim whitespace and convert to lowercase
        normalized = normalized.strip().lower()
        
        return normalized
    
    def _extract_alternative_names(self, name: str) -> List[str]:
        """
        Extract alternative district names from parenthetical text.
        
        For names like "Keonjhar (Kendujhar)", returns both "keonjhar" and "kendujhar".
        
        Args:
            name: Original district name
            
        Returns:
            List of normalized alternative names
        """
        if pd.isna(name) or not isinstance(name, str):
            return []
        
        alternatives = []
        
        # Extract main name (without parentheses)
        main_name = re.sub(r'\s*\([^)]*\)', '', name).strip().lower()
        if main_name:
            alternatives.append(main_name)
        
        # Extract text inside parentheses
        parenthetical_matches = re.findall(r'\(([^)]+)\)', name)
        for match in parenthetical_matches:
            alt_name = match.strip().lower()
            if alt_name and alt_name not in alternatives:
                alternatives.append(alt_name)
        
        return alternatives
    
    def _build_district_lookup(self, lgd_df: pd.DataFrame) -> Dict[str, str]:
        """
        Build lookup dictionary from normalized district names to district codes.
        
        Args:
            lgd_df: LGD reference DataFrame with district and district_code columns
            
        Returns:
            Dict mapping normalized district name to district code
        """
        district_lookup = {}
        
        # Extract unique district name and code pairs
        district_data = lgd_df[['district', 'district_code']].drop_duplicates()
        
        for _, row in district_data.iterrows():
            district_name = row['district']
            district_code = str(row['district_code'])
            
            if pd.notna(district_name) and pd.notna(district_code):
                normalized_name = self._normalize_district_name(district_name)
                
                if normalized_name in district_lookup:
                    # Multiple LGD records for same district name
                    self.logger.info(
                        f"Multiple LGD records found for district '{district_name}', "
                        f"using first occurrence (code: {district_lookup[normalized_name]})"
                    )
                else:
                    district_lookup[normalized_name] = district_code
        
        self.logger.debug(f"Built district lookup with {len(district_lookup)} entries")
        
        return district_lookup
    
    def _find_district_code_exact(
        self, 
        normalized_name: str, 
        district_lookup: Dict[str, str]
    ) -> Optional[str]:
        """
        Find district code using exact normalized name match.
        
        Args:
            normalized_name: Normalized district name to search for
            district_lookup: Dictionary mapping normalized names to codes
            
        Returns:
            District code if found, None otherwise
        """
        return district_lookup.get(normalized_name)
    
    def _find_district_code_fuzzy(
        self,
        normalized_name: str,
        district_lookup: Dict[str, str],
        threshold: int
    ) -> Optional[str]:
        """
        Find district code using fuzzy string matching.
        
        Args:
            normalized_name: Normalized district name to search for
            district_lookup: Dictionary mapping normalized names to codes
            threshold: Minimum similarity score (0-100) for a match
            
        Returns:
            District code if fuzzy match found above threshold, None otherwise
        """
        if not normalized_name:
            return None
        
        best_match = None
        best_score = 0
        
        for lgd_name, district_code in district_lookup.items():
            # Calculate similarity score
            score = fuzz.ratio(normalized_name, lgd_name)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = district_code
                self.logger.debug(
                    f"Fuzzy match candidate: '{normalized_name}' ~ '{lgd_name}' "
                    f"(score: {score})"
                )
        
        return best_match
    
    def _validate_entities_dataframe(self, entities_df: pd.DataFrame) -> None:
        """
        Validate entities DataFrame has required columns and is not empty.
        
        Args:
            entities_df: Entity DataFrame to validate
            
        Raises:
            ValidationError: If validation fails
        """
        # Check if DataFrame is empty
        if entities_df is None or len(entities_df) == 0:
            raise ValidationError(
                message="Cannot perform district code mapping on empty entities DataFrame",
                field_name="entities_df",
                invalid_value="empty",
                validation_rules=["DataFrame must contain at least one record"]
            )
        
        # Check for required 'district' column
        if 'district' not in entities_df.columns:
            raise ValidationError(
                message="Missing required column 'district' in entities DataFrame",
                field_name="district",
                invalid_value=f"Available columns: {list(entities_df.columns)}",
                validation_rules=["entities DataFrame must contain 'district' column"]
            )
        
        self.logger.debug("Entities DataFrame validation passed")
    
    def _validate_lgd_dataframe(self, lgd_df: pd.DataFrame) -> None:
        """
        Validate LGD DataFrame has required columns and is not empty.
        
        Args:
            lgd_df: LGD reference DataFrame to validate
            
        Raises:
            ValidationError: If validation fails
        """
        # Check if DataFrame is empty
        if lgd_df is None or len(lgd_df) == 0:
            raise ValidationError(
                message="Cannot perform district code mapping with empty LGD reference DataFrame",
                field_name="lgd_df",
                invalid_value="empty",
                validation_rules=["LGD DataFrame must contain at least one record"]
            )
        
        # Check for required 'district' column
        if 'district' not in lgd_df.columns:
            raise ValidationError(
                message="Missing required column 'district' in LGD reference DataFrame",
                field_name="district",
                invalid_value=f"Available columns: {list(lgd_df.columns)}",
                validation_rules=["LGD DataFrame must contain 'district' column"]
            )
        
        # Check for required 'district_code' column
        if 'district_code' not in lgd_df.columns:
            raise ValidationError(
                message="Missing required column 'district_code' in LGD reference DataFrame",
                field_name="district_code",
                invalid_value=f"Available columns: {list(lgd_df.columns)}",
                validation_rules=["LGD DataFrame must contain 'district_code' column"]
            )
        
        self.logger.debug("LGD DataFrame validation passed")
