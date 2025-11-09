"""
Data loading and validation module.

This module provides the DataLoader class for loading CSV files with comprehensive
validation and error handling.
"""

import pandas as pd
import os
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path

from .models import EntityRecord, LGDRecord
from .exceptions import DataLoadError, ValidationError, FileAccessError, DataQualityError
from .utils.data_utils import (
    clean_dataframe_strings,
    convert_numeric_columns,
    get_data_quality_summary,
    detect_duplicates
)
from .utils.error_handler import (
    ErrorHandler, RetryConfig, safe_file_operation, with_retry, 
    create_error_context, log_error_details
)
from .utils.data_validator import DataValidator, DataQualityReport


class DataLoader:
    """
    Handles loading and validation of CSV files for the LGD mapping process.
    
    This class provides methods to load entity and LGD code data from CSV files
    with comprehensive validation and error handling.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, 
                 error_handler: Optional[ErrorHandler] = None,
                 retry_config: Optional[RetryConfig] = None,
                 data_validator: Optional[DataValidator] = None):
        """
        Initialize the DataLoader.
        
        Args:
            logger: Optional logger instance for logging operations
            error_handler: Optional error handler for recovery strategies
            retry_config: Optional retry configuration for file operations
            data_validator: Optional data validator for quality checks
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = error_handler or ErrorHandler(self.logger)
        self.retry_config = retry_config or RetryConfig(max_attempts=3, base_delay=1.0)
        self.data_validator = data_validator or DataValidator(self.logger)
    
    def load_entities(self, file_path: str) -> pd.DataFrame:
        """
        Load entity data from CSV file with validation.
        
        Args:
            file_path: Path to the entities CSV file
            
        Returns:
            DataFrame containing validated entity data
            
        Raises:
            DataLoadError: If file cannot be loaded or validation fails
        """
        self.logger.info(f"Loading entities from: {file_path}")
        
        try:
            # Check file existence and accessibility
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileAccessError(
                    f"Entities file not found: {file_path}",
                    file_path=file_path,
                    operation="read"
                )
            
            if not file_path_obj.is_file():
                raise FileAccessError(
                    f"Path is not a file: {file_path}",
                    file_path=file_path,
                    operation="read"
                )
            
            # Load CSV file with retry mechanism
            def load_csv():
                return pd.read_csv(file_path)
            
            df = safe_file_operation(
                operation=load_csv,
                file_path=file_path,
                operation_name="read CSV",
                retry_config=self.retry_config,
                logger=self.logger
            )
            
            self.logger.info(f"Loaded {len(df)} entity records")
            
            # Validate file is not empty
            if df.empty:
                raise DataLoadError(
                    "Entities file contains no data",
                    file_path=file_path
                )
            
            # Validate required columns
            required_columns = ['district', 'block', 'village']
            self._validate_columns(df, required_columns, 'entities', file_path)
            
            # Clean and process data with error handling
            df = self._process_entity_data(df, file_path)
            
            # Validate data quality
            self._validate_entity_data(df, file_path)
            
            # Log data quality summary
            quality_summary = get_data_quality_summary(df)
            self.logger.info(f"Entity data quality: {quality_summary}")
            
            return df
            
        except (FileAccessError, DataLoadError, ValidationError):
            # Re-raise our custom exceptions
            raise
        except pd.errors.EmptyDataError as e:
            raise DataLoadError(
                "Entities file is empty or contains no valid data",
                file_path=file_path,
                original_error=e
            )
        except pd.errors.ParserError as e:
            raise DataLoadError(
                f"Error parsing entities CSV file: {str(e)}",
                file_path=file_path,
                original_error=e
            )
        except PermissionError as e:
            raise FileAccessError(
                f"Permission denied accessing entities file: {file_path}",
                file_path=file_path,
                operation="read",
                original_error=e
            )
        except Exception as e:
            # Log unexpected errors with full context
            context = create_error_context(
                operation="load_entities",
                file_path=file_path,
                error_type=type(e).__name__
            )
            log_error_details(self.logger, e, context)
            
            raise DataLoadError(
                f"Unexpected error loading entities from {file_path}: {str(e)}",
                file_path=file_path,
                original_error=e
            )
    
    def load_lgd_codes(self, file_path: str) -> pd.DataFrame:
        """
        Load LGD codes data from CSV file with validation.
        
        Args:
            file_path: Path to the LGD codes CSV file
            
        Returns:
            DataFrame containing validated LGD codes data
            
        Raises:
            DataLoadError: If file cannot be loaded or validation fails
        """
        self.logger.info(f"Loading LGD codes from: {file_path}")
        
        try:
            # Check file existence and accessibility
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileAccessError(
                    f"LGD codes file not found: {file_path}",
                    file_path=file_path,
                    operation="read"
                )
            
            if not file_path_obj.is_file():
                raise FileAccessError(
                    f"Path is not a file: {file_path}",
                    file_path=file_path,
                    operation="read"
                )
            
            # Load CSV file with retry mechanism
            def load_csv():
                return pd.read_csv(file_path)
            
            df = safe_file_operation(
                operation=load_csv,
                file_path=file_path,
                operation_name="read CSV",
                retry_config=self.retry_config,
                logger=self.logger
            )
            
            self.logger.info(f"Loaded {len(df)} LGD code records")
            
            # Validate file is not empty
            if df.empty:
                raise DataLoadError(
                    "LGD codes file contains no data",
                    file_path=file_path
                )
            
            # Validate required columns
            required_columns = [
                'district_code', 'district', 'block_code', 'block',
                'village_code', 'village'
            ]
            self._validate_columns(df, required_columns, 'LGD codes', file_path)
            
            # Clean and process data with error handling
            df = self._process_lgd_data(df, file_path)
            
            # Validate data quality
            self._validate_lgd_data(df, file_path)
            
            # Log data quality summary
            quality_summary = get_data_quality_summary(df)
            self.logger.info(f"LGD codes data quality: {quality_summary}")
            
            return df
            
        except (FileAccessError, DataLoadError, ValidationError):
            # Re-raise our custom exceptions
            raise
        except pd.errors.EmptyDataError as e:
            raise DataLoadError(
                "LGD codes file is empty or contains no valid data",
                file_path=file_path,
                original_error=e
            )
        except pd.errors.ParserError as e:
            raise DataLoadError(
                f"Error parsing LGD codes CSV file: {str(e)}",
                file_path=file_path,
                original_error=e
            )
        except PermissionError as e:
            raise FileAccessError(
                f"Permission denied accessing LGD codes file: {file_path}",
                file_path=file_path,
                operation="read",
                original_error=e
            )
        except Exception as e:
            # Log unexpected errors with full context
            context = create_error_context(
                operation="load_lgd_codes",
                file_path=file_path,
                error_type=type(e).__name__
            )
            log_error_details(self.logger, e, context)
            
            raise DataLoadError(
                f"Unexpected error loading LGD codes from {file_path}: {str(e)}",
                file_path=file_path,
                original_error=e
            )
    
    def validate_file_format(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """
        Validate that DataFrame has expected columns.
        
        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            
        Returns:
            True if all expected columns are present, False otherwise
        """
        missing_columns = set(expected_columns) - set(df.columns)
        return len(missing_columns) == 0
    
    def _validate_columns(self, df: pd.DataFrame, required_columns: List[str], 
                         data_type: str, file_path: str) -> None:
        """
        Validate that required columns are present in DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            data_type: Type of data being validated (for error messages)
            file_path: Path to the file being validated
            
        Raises:
            ValidationError: If required columns are missing
        """
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            available_columns = list(df.columns)
            raise ValidationError(
                f"Missing required columns in {data_type} file: {sorted(missing_columns)}. "
                f"Available columns: {sorted(available_columns)}",
                field_name="columns",
                invalid_value=available_columns,
                validation_rules=[f"Must contain columns: {sorted(required_columns)}"]
            )
    
    def _process_entity_data(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """
        Process and clean entity data with error handling.
        Handles all hierarchical fields (state, district, block, subdistrict, gp, village).
        
        Args:
            df: Raw entity DataFrame
            file_path: Path to the source file
            
        Returns:
            Processed DataFrame
        """
        try:
            # Define all possible hierarchical columns
            hierarchical_name_columns = ['state', 'district', 'block', 'subdistrict', 'gp', 'village']
            hierarchical_code_columns = ['state_code', 'district_code', 'block_code', 'subdistrict_code', 'gp_code', 'village_code']
            
            # Clean string columns that are present
            string_columns = [col for col in hierarchical_name_columns if col in df.columns]
            if string_columns:
                df = clean_dataframe_strings(df, string_columns)
            
            # Process code columns - add if not present, convert to int if present
            for code_col in hierarchical_code_columns:
                if code_col not in df.columns:
                    df[code_col] = None
                else:
                    # Convert code to int if present
                    try:
                        numeric_columns = {code_col: 'int'}
                        df = convert_numeric_columns(df, numeric_columns)
                    except Exception as e:
                        self.logger.warning(f"Error converting {code_col} to numeric: {e}")
                        # Continue with code as object type
            
            return df
            
        except Exception as e:
            raise DataLoadError(
                f"Error processing entity data: {str(e)}",
                file_path=file_path,
                original_error=e
            )
    
    def _process_lgd_data(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """
        Process and clean LGD codes data with error handling.
        Handles all hierarchical fields (state, district, block, subdistrict, gp, village).
        
        Args:
            df: Raw LGD codes DataFrame
            file_path: Path to the source file
            
        Returns:
            Processed DataFrame
        """
        try:
            # Define all possible hierarchical columns
            hierarchical_name_columns = ['state', 'district', 'block', 'subdistrict', 'gp', 'village']
            hierarchical_code_columns = ['state_code', 'district_code', 'block_code', 'subdistrict_code', 'gp_code', 'village_code']
            
            # Clean string columns that are present
            string_columns = [col for col in hierarchical_name_columns if col in df.columns]
            if string_columns:
                df = clean_dataframe_strings(df, string_columns)
            
            # Convert numeric columns that are present
            numeric_columns = {}
            for code_col in hierarchical_code_columns:
                if code_col in df.columns:
                    numeric_columns[code_col] = 'int'
            
            if numeric_columns:
                try:
                    df = convert_numeric_columns(df, numeric_columns)
                except Exception as e:
                    self.logger.warning(f"Error converting numeric columns: {e}")
                    # Try to convert individual columns
                    for col, dtype in numeric_columns.items():
                        if col in df.columns:
                            try:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            except Exception as col_error:
                                self.logger.warning(f"Error converting column {col}: {col_error}")
            
            return df
            
        except Exception as e:
            raise DataLoadError(
                f"Error processing LGD codes data: {str(e)}",
                file_path=file_path,
                original_error=e
            )
    
    def _validate_entity_data(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Validate entity data quality and log warnings for issues.
        Includes hierarchical consistency checks.
        
        Args:
            df: Entity DataFrame to validate
            file_path: Path to the source file
        """
        try:
            quality_issues = []
            
            # Check for completely empty records
            empty_records = df[
                (df['district'].isna() | (df['district'].str.strip() == '')) &
                (df['block'].isna() | (df['block'].str.strip() == '')) &
                (df['village'].isna() | (df['village'].str.strip() == ''))
            ]
            
            if len(empty_records) > 0:
                issue = f"Found {len(empty_records)} completely empty entity records"
                self.logger.warning(issue)
                quality_issues.append(issue)
            
            # Check for records with missing required fields
            missing_district = df[df['district'].isna() | (df['district'].str.strip() == '')]
            missing_block = df[df['block'].isna() | (df['block'].str.strip() == '')]
            missing_village = df[df['village'].isna() | (df['village'].str.strip() == '')]
            
            if len(missing_district) > 0:
                issue = f"Found {len(missing_district)} records with missing district"
                self.logger.warning(issue)
                quality_issues.append(issue)
            if len(missing_block) > 0:
                issue = f"Found {len(missing_block)} records with missing block"
                self.logger.warning(issue)
                quality_issues.append(issue)
            if len(missing_village) > 0:
                issue = f"Found {len(missing_village)} records with missing village"
                self.logger.warning(issue)
                quality_issues.append(issue)
            
            # Check for duplicates
            try:
                duplicates = detect_duplicates(df, ['district', 'block', 'village'])
                if len(duplicates) > 0:
                    issue = f"Found {len(duplicates)} duplicate entity records"
                    self.logger.warning(issue)
                    quality_issues.append(issue)
            except Exception as e:
                self.logger.warning(f"Error checking for duplicates: {e}")
            
            # NEW: Validate hierarchical consistency in entity data
            try:
                self._validate_entity_hierarchical_consistency(df)
            except Exception as e:
                self.logger.warning(f"Error validating entity hierarchical consistency: {e}")
            
            # Raise data quality error if critical issues found
            critical_issues = len(empty_records) + len(missing_district) + len(missing_block) + len(missing_village)
            if critical_issues > len(df) * 0.5:  # More than 50% of records have critical issues
                raise DataQualityError(
                    f"Critical data quality issues in entity file: {critical_issues}/{len(df)} records affected",
                    quality_issue="high_missing_data_rate",
                    affected_records=critical_issues,
                    severity="high",
                    recommendations=[
                        "Review source data quality",
                        "Check data extraction process",
                        "Validate required field mappings"
                    ]
                )
                
        except DataQualityError:
            raise
        except Exception as e:
            self.logger.warning(f"Error during entity data validation: {e}")
            # Don't fail the entire process for validation errors
    
    def _validate_lgd_data(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Validate LGD codes data quality and log warnings for issues.
        Includes validation for all hierarchical levels.
        
        Args:
            df: LGD codes DataFrame to validate
            file_path: Path to the source file
        """
        try:
            quality_issues = []
            critical_issues_count = 0
            
            # Check for records with null codes at all hierarchical levels
            hierarchical_code_columns = ['state_code', 'district_code', 'block_code', 'subdistrict_code', 'gp_code', 'village_code']
            
            for code_col in hierarchical_code_columns:
                if code_col in df.columns:
                    null_codes = df[df[code_col].isna()]
                    if len(null_codes) > 0:
                        issue = f"Found {len(null_codes)} records with null {code_col}"
                        self.logger.warning(issue)
                        quality_issues.append(issue)
                        # Only count required fields as critical
                        if code_col in ['district_code', 'block_code', 'village_code']:
                            critical_issues_count += len(null_codes)
            
            # Check for records with missing names at all hierarchical levels
            hierarchical_name_columns = ['state', 'district', 'block', 'subdistrict', 'gp', 'village']
            
            for name_col in hierarchical_name_columns:
                if name_col in df.columns:
                    missing_names = df[df[name_col].isna() | (df[name_col].str.strip() == '')]
                    if len(missing_names) > 0:
                        issue = f"Found {len(missing_names)} records with missing {name_col} names"
                        self.logger.warning(issue)
                        quality_issues.append(issue)
                        # Only count required fields as critical
                        if name_col in ['district', 'block', 'village']:
                            critical_issues_count += len(missing_names)
            
            # Check for duplicate codes
            try:
                duplicate_village_codes = detect_duplicates(df, ['village_code'])
                if len(duplicate_village_codes) > 0:
                    issue = f"Found {len(duplicate_village_codes)} records with duplicate village codes"
                    self.logger.warning(issue)
                    quality_issues.append(issue)
            except Exception as e:
                self.logger.warning(f"Error checking for duplicate village codes: {e}")
            
            # Check for inconsistent hierarchical data
            try:
                self._validate_hierarchical_consistency(df)
            except Exception as e:
                self.logger.warning(f"Error validating hierarchical consistency: {e}")
            
            # NEW: Check hierarchical parent-child relationships
            try:
                self._validate_hierarchical_relationships(df)
            except Exception as e:
                self.logger.warning(f"Error validating hierarchical relationships: {e}")
            
            # Raise data quality error if critical issues found
            if critical_issues_count > len(df) * 0.3:  # More than 30% of records have critical issues
                raise DataQualityError(
                    f"Critical data quality issues in LGD codes file: {critical_issues_count}/{len(df)} records affected",
                    quality_issue="high_missing_reference_data_rate",
                    affected_records=critical_issues_count,
                    severity="high",
                    recommendations=[
                        "Review LGD reference data completeness",
                        "Check data source integrity",
                        "Validate code assignment process"
                    ]
                )
                
        except DataQualityError:
            raise
        except Exception as e:
            self.logger.warning(f"Error during LGD data validation: {e}")
            # Don't fail the entire process for validation errors
    
    def _validate_hierarchical_relationships(self, df: pd.DataFrame) -> None:
        """
        Validate parent-child relationships in hierarchical data.
        Ensures that child entities belong to their specified parents.
        
        Args:
            df: LGD DataFrame to validate
        """
        try:
            # Validate district belongs to state (if state present)
            if 'state_code' in df.columns and 'district_code' in df.columns:
                # Check if each district_code appears under only one state_code
                district_state_mapping = df.groupby('district_code')['state_code'].nunique()
                multi_state_districts = district_state_mapping[district_state_mapping > 1]
                
                if len(multi_state_districts) > 0:
                    self.logger.warning(
                        f"Found {len(multi_state_districts)} district codes appearing under multiple states"
                    )
                    for district_code in list(multi_state_districts.index)[:5]:
                        states = df[df['district_code'] == district_code]['state_code'].unique()
                        self.logger.debug(f"District code {district_code} appears in states: {list(states)}")
            
            # Validate block belongs to district
            if 'district_code' in df.columns and 'block_code' in df.columns:
                # Check if each block_code appears under only one district_code
                block_district_mapping = df.groupby('block_code')['district_code'].nunique()
                multi_district_blocks = block_district_mapping[block_district_mapping > 1]
                
                if len(multi_district_blocks) > 0:
                    self.logger.warning(
                        f"Found {len(multi_district_blocks)} block codes appearing under multiple districts"
                    )
                    for block_code in list(multi_district_blocks.index)[:5]:
                        districts = df[df['block_code'] == block_code]['district_code'].unique()
                        self.logger.debug(f"Block code {block_code} appears in districts: {list(districts)}")
            
            # Validate subdistrict belongs to district (if present)
            if 'district_code' in df.columns and 'subdistrict_code' in df.columns:
                subdistrict_district_mapping = df.groupby('subdistrict_code')['district_code'].nunique()
                multi_district_subdistricts = subdistrict_district_mapping[subdistrict_district_mapping > 1]
                
                if len(multi_district_subdistricts) > 0:
                    self.logger.warning(
                        f"Found {len(multi_district_subdistricts)} subdistrict codes appearing under multiple districts"
                    )
                    for subdistrict_code in list(multi_district_subdistricts.index)[:5]:
                        districts = df[df['subdistrict_code'] == subdistrict_code]['district_code'].unique()
                        self.logger.debug(f"Subdistrict code {subdistrict_code} appears in districts: {list(districts)}")
            
            # Validate GP belongs to block (if present)
            if 'block_code' in df.columns and 'gp_code' in df.columns:
                gp_block_mapping = df.groupby('gp_code')['block_code'].nunique()
                multi_block_gps = gp_block_mapping[gp_block_mapping > 1]
                
                if len(multi_block_gps) > 0:
                    self.logger.warning(
                        f"Found {len(multi_block_gps)} GP codes appearing under multiple blocks"
                    )
                    for gp_code in list(multi_block_gps.index)[:5]:
                        blocks = df[df['gp_code'] == gp_code]['block_code'].unique()
                        self.logger.debug(f"GP code {gp_code} appears in blocks: {list(blocks)}")
            
            # Validate village belongs to GP or block
            if 'village_code' in df.columns:
                if 'gp_code' in df.columns:
                    # Village should belong to only one GP
                    village_gp_mapping = df.groupby('village_code')['gp_code'].nunique()
                    multi_gp_villages = village_gp_mapping[village_gp_mapping > 1]
                    
                    if len(multi_gp_villages) > 0:
                        self.logger.warning(
                            f"Found {len(multi_gp_villages)} village codes appearing under multiple GPs"
                        )
                        for village_code in list(multi_gp_villages.index)[:5]:
                            gps = df[df['village_code'] == village_code]['gp_code'].unique()
                            self.logger.debug(f"Village code {village_code} appears in GPs: {list(gps)}")
                elif 'block_code' in df.columns:
                    # Village should belong to only one block
                    village_block_mapping = df.groupby('village_code')['block_code'].nunique()
                    multi_block_villages = village_block_mapping[village_block_mapping > 1]
                    
                    if len(multi_block_villages) > 0:
                        self.logger.warning(
                            f"Found {len(multi_block_villages)} village codes appearing under multiple blocks"
                        )
                        for village_code in list(multi_block_villages.index)[:5]:
                            blocks = df[df['village_code'] == village_code]['block_code'].unique()
                            self.logger.debug(f"Village code {village_code} appears in blocks: {list(blocks)}")
                    
        except Exception as e:
            self.logger.warning(f"Error validating hierarchical relationships: {e}")
            # Don't fail validation for this check
    
    def _validate_entity_hierarchical_consistency(self, df: pd.DataFrame) -> None:
        """
        Validate hierarchical consistency in entity data.
        Checks that hierarchical relationships are consistent.
        
        Args:
            df: Entity DataFrame to validate
        """
        try:
            # Check if same state has consistent state_code (if both present)
            if 'state' in df.columns and 'state_code' in df.columns:
                state_with_codes = df[df['state_code'].notna() & df['state'].notna()]
                if len(state_with_codes) > 0:
                    state_inconsistencies = state_with_codes.groupby('state')['state_code'].nunique()
                    inconsistent_states = state_inconsistencies[state_inconsistencies > 1]
                    
                    if len(inconsistent_states) > 0:
                        self.logger.warning(
                            f"Found {len(inconsistent_states)} state names with multiple state codes"
                        )
                        for state_name in list(inconsistent_states.index)[:5]:
                            codes = state_with_codes[state_with_codes['state'] == state_name]['state_code'].unique()
                            self.logger.debug(f"State '{state_name}' has codes: {list(codes)}")
            
            # Check if same district has consistent district_code (if both present)
            if 'district' in df.columns and 'district_code' in df.columns:
                district_with_codes = df[df['district_code'].notna() & df['district'].notna()]
                if len(district_with_codes) > 0:
                    # Check within state if state is present
                    if 'state' in df.columns:
                        district_groups = district_with_codes.groupby(['state', 'district'])['district_code'].nunique()
                        inconsistent_districts = district_groups[district_groups > 1]
                        
                        if len(inconsistent_districts) > 0:
                            self.logger.warning(
                                f"Found {len(inconsistent_districts)} district names with multiple district codes within states"
                            )
                            for (state, district) in list(inconsistent_districts.index)[:5]:
                                codes = district_with_codes[(district_with_codes['state'] == state) & 
                                                           (district_with_codes['district'] == district)]['district_code'].unique()
                                self.logger.debug(f"District '{district}' in state '{state}' has codes: {list(codes)}")
                    else:
                        district_inconsistencies = district_with_codes.groupby('district')['district_code'].nunique()
                        inconsistent_districts = district_inconsistencies[district_inconsistencies > 1]
                        
                        if len(inconsistent_districts) > 0:
                            self.logger.warning(
                                f"Found {len(inconsistent_districts)} district names with multiple district codes"
                            )
                            for district_name in list(inconsistent_districts.index)[:5]:
                                codes = district_with_codes[district_with_codes['district'] == district_name]['district_code'].unique()
                                self.logger.debug(f"District '{district_name}' has codes: {list(codes)}")
            
            # Check if same block has consistent block_code (if both present)
            if 'block' in df.columns and 'block_code' in df.columns:
                block_with_codes = df[df['block_code'].notna() & df['block'].notna() & df['district'].notna()]
                if len(block_with_codes) > 0:
                    block_groups = block_with_codes.groupby(['district', 'block'])['block_code'].nunique()
                    inconsistent_blocks = block_groups[block_groups > 1]
                    
                    if len(inconsistent_blocks) > 0:
                        self.logger.warning(
                            f"Found {len(inconsistent_blocks)} block names with multiple block codes within districts"
                        )
                        for (district, block) in list(inconsistent_blocks.index)[:5]:
                            codes = block_with_codes[(block_with_codes['district'] == district) & 
                                                    (block_with_codes['block'] == block)]['block_code'].unique()
                            self.logger.debug(f"Block '{block}' in district '{district}' has codes: {list(codes)}")
            
            # Check if same GP has consistent gp_code (if both present)
            if 'gp' in df.columns and 'gp_code' in df.columns:
                gp_with_codes = df[df['gp_code'].notna() & df['gp'].notna() & df['block'].notna()]
                if len(gp_with_codes) > 0:
                    gp_groups = gp_with_codes.groupby(['block', 'gp'])['gp_code'].nunique()
                    inconsistent_gps = gp_groups[gp_groups > 1]
                    
                    if len(inconsistent_gps) > 0:
                        self.logger.warning(
                            f"Found {len(inconsistent_gps)} GP names with multiple GP codes within blocks"
                        )
                        for (block, gp) in list(inconsistent_gps.index)[:5]:
                            codes = gp_with_codes[(gp_with_codes['block'] == block) & 
                                                 (gp_with_codes['gp'] == gp)]['gp_code'].unique()
                            self.logger.debug(f"GP '{gp}' in block '{block}' has codes: {list(codes)}")
                    
        except Exception as e:
            self.logger.warning(f"Error validating entity hierarchical consistency: {e}")
            # Don't fail validation for this check
    
    def _validate_hierarchical_consistency(self, df: pd.DataFrame) -> None:
        """
        Validate hierarchical consistency in LGD data.
        Checks all hierarchical levels (state, district, block, subdistrict, gp, village).
        
        Args:
            df: LGD codes DataFrame to validate
        """
        try:
            # Check state level consistency (if present)
            if 'state_code' in df.columns and 'state' in df.columns:
                state_inconsistencies = df.groupby('state_code')['state'].nunique()
                inconsistent_states = state_inconsistencies[state_inconsistencies > 1]
                
                if len(inconsistent_states) > 0:
                    self.logger.warning(
                        f"Found {len(inconsistent_states)} state codes with inconsistent names"
                    )
                    for state_code in inconsistent_states.index[:5]:
                        names = df[df['state_code'] == state_code]['state'].unique()
                        self.logger.debug(f"State code {state_code} has names: {list(names)}")
            
            # Check district level consistency
            if 'district_code' in df.columns and 'district' in df.columns:
                # Check within state if state is present
                if 'state_code' in df.columns:
                    district_groups = df.groupby(['state_code', 'district_code'])['district'].nunique()
                    inconsistent_districts = district_groups[district_groups > 1]
                    
                    if len(inconsistent_districts) > 0:
                        self.logger.warning(
                            f"Found {len(inconsistent_districts)} district codes with inconsistent names within states"
                        )
                        for (state_code, district_code) in list(inconsistent_districts.index)[:5]:
                            names = df[(df['state_code'] == state_code) & 
                                      (df['district_code'] == district_code)]['district'].unique()
                            self.logger.debug(f"District code {district_code} in state {state_code} has names: {list(names)}")
                else:
                    district_inconsistencies = df.groupby('district_code')['district'].nunique()
                    inconsistent_districts = district_inconsistencies[district_inconsistencies > 1]
                    
                    if len(inconsistent_districts) > 0:
                        self.logger.warning(
                            f"Found {len(inconsistent_districts)} district codes with inconsistent names"
                        )
                        for district_code in inconsistent_districts.index[:5]:
                            names = df[df['district_code'] == district_code]['district'].unique()
                            self.logger.debug(f"District code {district_code} has names: {list(names)}")
            
            # Check block level consistency
            if 'block_code' in df.columns and 'block' in df.columns:
                block_groups = df.groupby(['district_code', 'block_code'])['block'].nunique()
                inconsistent_blocks = block_groups[block_groups > 1]
                
                if len(inconsistent_blocks) > 0:
                    self.logger.warning(
                        f"Found {len(inconsistent_blocks)} block codes with inconsistent names"
                    )
                    for (district_code, block_code) in list(inconsistent_blocks.index)[:5]:
                        names = df[(df['district_code'] == district_code) & 
                                  (df['block_code'] == block_code)]['block'].unique()
                        self.logger.debug(f"Block code {block_code} in district {district_code} has names: {list(names)}")
            
            # Check subdistrict level consistency (if present)
            if 'subdistrict_code' in df.columns and 'subdistrict' in df.columns:
                subdistrict_groups = df.groupby(['district_code', 'subdistrict_code'])['subdistrict'].nunique()
                inconsistent_subdistricts = subdistrict_groups[subdistrict_groups > 1]
                
                if len(inconsistent_subdistricts) > 0:
                    self.logger.warning(
                        f"Found {len(inconsistent_subdistricts)} subdistrict codes with inconsistent names"
                    )
                    for (district_code, subdistrict_code) in list(inconsistent_subdistricts.index)[:5]:
                        names = df[(df['district_code'] == district_code) & 
                                  (df['subdistrict_code'] == subdistrict_code)]['subdistrict'].unique()
                        self.logger.debug(f"Subdistrict code {subdistrict_code} in district {district_code} has names: {list(names)}")
            
            # Check GP level consistency (if present)
            if 'gp_code' in df.columns and 'gp' in df.columns:
                # GP should be unique within block
                gp_groups = df.groupby(['block_code', 'gp_code'])['gp'].nunique()
                inconsistent_gps = gp_groups[gp_groups > 1]
                
                if len(inconsistent_gps) > 0:
                    self.logger.warning(
                        f"Found {len(inconsistent_gps)} GP codes with inconsistent names"
                    )
                    for (block_code, gp_code) in list(inconsistent_gps.index)[:5]:
                        names = df[(df['block_code'] == block_code) & 
                                  (df['gp_code'] == gp_code)]['gp'].unique()
                        self.logger.debug(f"GP code {gp_code} in block {block_code} has names: {list(names)}")
            
            # Check village level consistency
            if 'village_code' in df.columns and 'village' in df.columns:
                # Village should be unique within GP (if GP present) or block
                if 'gp_code' in df.columns:
                    village_groups = df.groupby(['gp_code', 'village_code'])['village'].nunique()
                    inconsistent_villages = village_groups[village_groups > 1]
                    
                    if len(inconsistent_villages) > 0:
                        self.logger.warning(
                            f"Found {len(inconsistent_villages)} village codes with inconsistent names within GPs"
                        )
                        for (gp_code, village_code) in list(inconsistent_villages.index)[:5]:
                            names = df[(df['gp_code'] == gp_code) & 
                                      (df['village_code'] == village_code)]['village'].unique()
                            self.logger.debug(f"Village code {village_code} in GP {gp_code} has names: {list(names)}")
                else:
                    village_groups = df.groupby(['block_code', 'village_code'])['village'].nunique()
                    inconsistent_villages = village_groups[village_groups > 1]
                    
                    if len(inconsistent_villages) > 0:
                        self.logger.warning(
                            f"Found {len(inconsistent_villages)} village codes with inconsistent names within blocks"
                        )
                        for (block_code, village_code) in list(inconsistent_villages.index)[:5]:
                            names = df[(df['block_code'] == block_code) & 
                                      (df['village_code'] == village_code)]['village'].unique()
                            self.logger.debug(f"Village code {village_code} in block {block_code} has names: {list(names)}")
                    
        except Exception as e:
            self.logger.warning(f"Error validating hierarchical consistency: {e}")
            # Don't fail validation for this check
    
    def create_entity_records(self, df: pd.DataFrame) -> List[EntityRecord]:
        """
        Convert DataFrame to list of EntityRecord objects.
        
        Args:
            df: DataFrame containing entity data
            
        Returns:
            List of EntityRecord objects
        """
        records = []
        for _, row in df.iterrows():
            try:
                record = EntityRecord(
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
                records.append(record)
            except Exception as e:
                self.logger.warning(f"Error creating EntityRecord: {e}")
        
        return records
    
    def create_lgd_records(self, df: pd.DataFrame) -> List[LGDRecord]:
        """
        Convert DataFrame to list of LGDRecord objects.
        
        Args:
            df: DataFrame containing LGD codes data
            
        Returns:
            List of LGDRecord objects
        """
        records = []
        error_count = 0
        
        for idx, row in df.iterrows():
            try:
                record = LGDRecord(
                    district_code=row['district_code'],
                    district=row['district'],
                    block_code=row['block_code'],
                    block=row['block'],
                    village_code=row['village_code'],
                    village=row['village'],
                    gp_code=row.get('gp_code'),
                    gp=row.get('gp')
                )
                
                # Validate the record
                if not record.is_valid():
                    validation_errors = record.get_validation_errors()
                    self.logger.warning(f"Invalid LGDRecord at row {idx}: {validation_errors}")
                    error_count += 1
                    continue
                
                records.append(record)
                
            except Exception as e:
                error_count += 1
                context = create_error_context(
                    operation="create_lgd_record",
                    row_index=idx,
                    row_data=row.to_dict()
                )
                log_error_details(self.logger, e, context)
                
                # Try to handle the error gracefully
                try:
                    result = self.error_handler.handle_error(
                        e, 
                        context, 
                        recovery_strategy='skip_record'
                    )
                    if result is not None:
                        records.append(result)
                except Exception as recovery_error:
                    self.logger.debug(f"Recovery failed for LGDRecord at row {idx}: {recovery_error}")
        
        if error_count > 0:
            self.logger.warning(f"Encountered {error_count} errors while creating LGDRecord objects")
            
            # Raise error if too many records failed
            if error_count > len(df) * 0.5:
                raise DataQualityError(
                    f"Failed to create LGDRecord objects for {error_count}/{len(df)} records",
                    quality_issue="high_record_creation_failure_rate",
                    affected_records=error_count,
                    severity="high"
                )
        
        self.logger.info(f"Successfully created {len(records)} LGDRecord objects")
        return records
    
    def validate_data_consistency(self, entities_df: pd.DataFrame, lgd_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate consistency between entity and LGD data.
        
        Args:
            entities_df: Entity DataFrame
            lgd_df: LGD codes DataFrame
            
        Returns:
            Dictionary with consistency validation results
        """
        try:
            consistency_report = {
                'validation_passed': True,
                'issues': [],
                'recommendations': []
            }
            
            # Check if entity districts exist in LGD data
            entity_districts = set(entities_df['district'].dropna().unique())
            lgd_districts = set(lgd_df['district'].dropna().unique())
            
            missing_districts = entity_districts - lgd_districts
            if missing_districts:
                issue = f"Entity districts not found in LGD data: {sorted(list(missing_districts))}"
                consistency_report['issues'].append(issue)
                consistency_report['recommendations'].append("Review district name mappings")
                self.logger.warning(issue)
            
            # Check district code coverage
            entities_with_codes = entities_df[entities_df['district_code'].notna()]
            if len(entities_with_codes) > 0:
                entity_district_codes = set(entities_with_codes['district_code'].unique())
                lgd_district_codes = set(lgd_df['district_code'].unique())
                
                invalid_codes = entity_district_codes - lgd_district_codes
                if invalid_codes:
                    issue = f"Entity district codes not found in LGD data: {sorted(list(invalid_codes))}"
                    consistency_report['issues'].append(issue)
                    consistency_report['recommendations'].append("Validate district code assignments")
                    self.logger.warning(issue)
            
            # Set overall validation status
            consistency_report['validation_passed'] = len(consistency_report['issues']) == 0
            
            return consistency_report
            
        except Exception as e:
            self.logger.warning(f"Error during data consistency validation: {e}")
            return {
                'validation_passed': False,
                'issues': [f"Validation error: {str(e)}"],
                'recommendations': ["Review data consistency validation process"]
            }
    
    def validate_entity_data_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Perform comprehensive quality validation on entity data.
        
        Args:
            df: Entity DataFrame to validate
            
        Returns:
            DataQualityReport with validation results
        """
        self.logger.info("Performing comprehensive entity data quality validation")
        
        try:
            report = self.data_validator.validate_entity_data(df)
            
            # Log quality score
            self.logger.info(f"Entity data quality score: {report.quality_score:.1f}/100")
            
            # Log critical issues
            if report.critical_issues > 0:
                self.logger.error(f"Found {report.critical_issues} critical data quality issues")
            if report.high_issues > 0:
                self.logger.warning(f"Found {report.high_issues} high-severity data quality issues")
            
            # Raise exception for critical quality issues
            if report.quality_score < 50:  # Below 50% quality score
                raise DataQualityError(
                    f"Entity data quality is critically low: {report.quality_score:.1f}/100",
                    quality_issue="critically_low_quality_score",
                    affected_records=len(df),
                    severity="critical",
                    recommendations=report.recommendations
                )
            
            return report
            
        except DataQualityError:
            raise
        except Exception as e:
            self.logger.error(f"Error during entity data quality validation: {e}")
            # Return a basic report indicating validation failure
            report = DataQualityReport(
                dataset_name="Entity Data",
                total_records=len(df)
            )
            report.quality_score = 0.0
            report.recommendations = ["Review data quality validation process"]
            return report
    
    def validate_lgd_data_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Perform comprehensive quality validation on LGD data.
        
        Args:
            df: LGD DataFrame to validate
            
        Returns:
            DataQualityReport with validation results
        """
        self.logger.info("Performing comprehensive LGD data quality validation")
        
        try:
            report = self.data_validator.validate_lgd_data(df)
            
            # Log quality score
            self.logger.info(f"LGD data quality score: {report.quality_score:.1f}/100")
            
            # Log critical issues
            if report.critical_issues > 0:
                self.logger.error(f"Found {report.critical_issues} critical LGD data quality issues")
            if report.high_issues > 0:
                self.logger.warning(f"Found {report.high_issues} high-severity LGD data quality issues")
            
            # Raise exception for critical quality issues
            if report.quality_score < 60:  # LGD data should have higher quality threshold
                raise DataQualityError(
                    f"LGD reference data quality is critically low: {report.quality_score:.1f}/100",
                    quality_issue="critically_low_reference_quality",
                    affected_records=len(df),
                    severity="critical",
                    recommendations=report.recommendations
                )
            
            return report
            
        except DataQualityError:
            raise
        except Exception as e:
            self.logger.error(f"Error during LGD data quality validation: {e}")
            # Return a basic report indicating validation failure
            report = DataQualityReport(
                dataset_name="LGD Reference Data",
                total_records=len(df)
            )
            report.quality_score = 0.0
            report.recommendations = ["Review LGD data quality validation process"]
            return report
    
    def detect_data_anomalies(self, df: pd.DataFrame, data_type: str) -> List[Dict[str, Any]]:
        """
        Detect anomalies and suspicious patterns in data.
        
        Args:
            df: DataFrame to analyze
            data_type: Type of data ('entity' or 'lgd')
            
        Returns:
            List of detected anomalies
        """
        self.logger.info(f"Detecting anomalies in {data_type} data")
        
        try:
            anomalies = self.data_validator.detect_anomalies(df, data_type)
            
            # Log anomaly summary
            if anomalies:
                self.logger.warning(f"Detected {len(anomalies)} anomalies in {data_type} data")
                
                # Log high-severity anomalies
                high_severity = [a for a in anomalies if a.get('severity') == 'high']
                if high_severity:
                    self.logger.error(f"Found {len(high_severity)} high-severity anomalies")
                    for anomaly in high_severity[:3]:  # Log first 3
                        self.logger.error(f"  - {anomaly.get('type', 'unknown')}: {anomaly.get('message', 'No message')}")
            else:
                self.logger.info(f"No anomalies detected in {data_type} data")
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error during anomaly detection: {e}")
            return [{
                'type': 'anomaly_detection_error',
                'message': f"Anomaly detection failed: {str(e)}",
                'severity': 'medium'
            }]
    
    def validate_cross_dataset_consistency(self, entities_df: pd.DataFrame, 
                                         lgd_df: pd.DataFrame) -> DataQualityReport:
        """
        Validate consistency between entity and LGD datasets.
        
        Args:
            entities_df: Entity DataFrame
            lgd_df: LGD DataFrame
            
        Returns:
            DataQualityReport with consistency validation results
        """
        self.logger.info("Validating cross-dataset consistency")
        
        try:
            report = self.data_validator.validate_data_consistency(entities_df, lgd_df)
            
            # Log consistency score
            self.logger.info(f"Cross-dataset consistency score: {report.quality_score:.1f}/100")
            
            # Log consistency issues
            if report.high_issues > 0:
                self.logger.warning(f"Found {report.high_issues} high-severity consistency issues")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error during consistency validation: {e}")
            # Return a basic report indicating validation failure
            report = DataQualityReport(
                dataset_name="Cross-Dataset Consistency",
                total_records=len(entities_df)
            )
            report.quality_score = 0.0
            report.recommendations = ["Review consistency validation process"]
            return report
    
    def generate_data_quality_summary(self, entity_report: DataQualityReport,
                                    lgd_report: DataQualityReport,
                                    consistency_report: Optional[DataQualityReport] = None) -> str:
        """
        Generate a comprehensive data quality summary.
        
        Args:
            entity_report: Entity data quality report
            lgd_report: LGD data quality report
            consistency_report: Optional consistency report
            
        Returns:
            Human-readable data quality summary
        """
        summary_lines = [
            "=== COMPREHENSIVE DATA QUALITY SUMMARY ===",
            "",
            f"Entity Data Quality: {entity_report.quality_score:.1f}/100",
            f"  - Critical Issues: {entity_report.critical_issues}",
            f"  - High Issues: {entity_report.high_issues}",
            f"  - Medium Issues: {entity_report.medium_issues}",
            f"  - Low Issues: {entity_report.low_issues}",
            "",
            f"LGD Reference Data Quality: {lgd_report.quality_score:.1f}/100",
            f"  - Critical Issues: {lgd_report.critical_issues}",
            f"  - High Issues: {lgd_report.high_issues}",
            f"  - Medium Issues: {lgd_report.medium_issues}",
            f"  - Low Issues: {lgd_report.low_issues}",
            ""
        ]
        
        if consistency_report:
            summary_lines.extend([
                f"Cross-Dataset Consistency: {consistency_report.quality_score:.1f}/100",
                f"  - Critical Issues: {consistency_report.critical_issues}",
                f"  - High Issues: {consistency_report.high_issues}",
                f"  - Medium Issues: {consistency_report.medium_issues}",
                f"  - Low Issues: {consistency_report.low_issues}",
                ""
            ])
        
        # Overall assessment
        overall_score = (entity_report.quality_score + lgd_report.quality_score) / 2
        if consistency_report:
            overall_score = (overall_score + consistency_report.quality_score) / 2
        
        summary_lines.extend([
            f"Overall Data Quality Score: {overall_score:.1f}/100",
            ""
        ])
        
        # Quality assessment
        if overall_score >= 90:
            assessment = "EXCELLENT - Data quality is very high"
        elif overall_score >= 80:
            assessment = "GOOD - Data quality is acceptable with minor issues"
        elif overall_score >= 70:
            assessment = "FAIR - Data quality needs improvement"
        elif overall_score >= 50:
            assessment = "POOR - Significant data quality issues present"
        else:
            assessment = "CRITICAL - Data quality is unacceptable"
        
        summary_lines.extend([
            f"Assessment: {assessment}",
            ""
        ])
        
        # Combined recommendations
        all_recommendations = set()
        all_recommendations.update(entity_report.recommendations)
        all_recommendations.update(lgd_report.recommendations)
        if consistency_report:
            all_recommendations.update(consistency_report.recommendations)
        
        if all_recommendations:
            summary_lines.extend([
                "Key Recommendations:",
                *[f"  - {rec}" for rec in sorted(all_recommendations)],
                ""
            ])
        
        return "\n".join(summary_lines)
    
    def perform_comprehensive_validation(self, entities_df: pd.DataFrame, 
                                       lgd_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive validation on both datasets.
        
        Args:
            entities_df: Entity DataFrame
            lgd_df: LGD DataFrame
            
        Returns:
            Dictionary with all validation results
        """
        self.logger.info("Starting comprehensive data validation")
        
        validation_results = {
            'entity_quality_report': None,
            'lgd_quality_report': None,
            'consistency_report': None,
            'entity_anomalies': [],
            'lgd_anomalies': [],
            'overall_quality_score': 0.0,
            'validation_summary': "",
            'critical_issues_found': False
        }
        
        try:
            # Validate entity data quality
            self.logger.info("Validating entity data quality...")
            validation_results['entity_quality_report'] = self.validate_entity_data_quality(entities_df)
            
            # Validate LGD data quality
            self.logger.info("Validating LGD data quality...")
            validation_results['lgd_quality_report'] = self.validate_lgd_data_quality(lgd_df)
            
            # Validate cross-dataset consistency
            self.logger.info("Validating cross-dataset consistency...")
            validation_results['consistency_report'] = self.validate_cross_dataset_consistency(entities_df, lgd_df)
            
            # Detect anomalies
            self.logger.info("Detecting anomalies...")
            validation_results['entity_anomalies'] = self.detect_data_anomalies(entities_df, 'entity')
            validation_results['lgd_anomalies'] = self.detect_data_anomalies(lgd_df, 'lgd')
            
            # Calculate overall quality score
            entity_score = validation_results['entity_quality_report'].quality_score
            lgd_score = validation_results['lgd_quality_report'].quality_score
            consistency_score = validation_results['consistency_report'].quality_score
            
            validation_results['overall_quality_score'] = (entity_score + lgd_score + consistency_score) / 3
            
            # Check for critical issues
            critical_issues = (
                validation_results['entity_quality_report'].critical_issues +
                validation_results['lgd_quality_report'].critical_issues +
                validation_results['consistency_report'].critical_issues
            )
            validation_results['critical_issues_found'] = critical_issues > 0
            
            # Generate summary
            validation_results['validation_summary'] = self.generate_data_quality_summary(
                validation_results['entity_quality_report'],
                validation_results['lgd_quality_report'],
                validation_results['consistency_report']
            )
            
            self.logger.info(f"Comprehensive validation completed. Overall score: {validation_results['overall_quality_score']:.1f}/100")
            
            if validation_results['critical_issues_found']:
                self.logger.error("Critical data quality issues found during validation")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive validation: {e}")
            validation_results['validation_summary'] = f"Validation failed: {str(e)}"
            return validation_results
    
    def get_loading_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the data loading process.
        
        Returns:
            Dictionary with loading statistics
        """
        error_summary = self.error_handler.get_error_summary()
        
        return {
            'error_handler_stats': error_summary,
            'retry_config': {
                'max_attempts': self.retry_config.max_attempts,
                'base_delay': self.retry_config.base_delay,
                'max_delay': self.retry_config.max_delay
            },
            'data_validator_available': self.data_validator is not None
        }