#!/usr/bin/env python3
"""
Input Data Validation Script

This script validates that your input CSV files have the correct format
and required columns before running the LGD mapping application.

Usage:
    python examples/validate_input.py --entities entities.csv --codes codes.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Validates input CSV files for LGD mapping."""
    
    # Required columns for each file type
    # District is always required, at least one lower level must be present
    ENTITIES_REQUIRED_COLUMNS = ['district']
    ENTITIES_LOWER_LEVEL_COLUMNS = ['block', 'gp', 'village']
    
    # District code and name are always required, at least one lower level pair must be present
    CODES_REQUIRED_COLUMNS = ['district_code', 'district']
    CODES_LOWER_LEVEL_PAIRS = [
        ('block_code', 'block'),
        ('gp_code', 'gp'),
        ('village_code', 'village')
    ]
    CODES_OPTIONAL_COLUMNS = ['state_code', 'state', 'subdistrict_code', 'subdistrict']
    
    def __init__(self, verbose=False):
        """Initialize validator."""
        self.verbose = verbose
        self.errors = []
        self.warnings = []
    
    def log(self, message, level='INFO'):
        """Log a message."""
        if self.verbose or level in ['ERROR', 'WARNING']:
            prefix = {
                'INFO': '✓',
                'WARNING': '⚠',
                'ERROR': '✗'
            }.get(level, ' ')
            print(f"{prefix} {message}")
    
    def validate_file_exists(self, file_path: str, file_type: str) -> bool:
        """Validate that file exists and is readable."""
        self.log(f"Checking {file_type} file: {file_path}")
        
        if not Path(file_path).exists():
            self.errors.append(f"{file_type} file not found: {file_path}")
            self.log(f"{file_type} file not found", 'ERROR')
            return False
        
        if not Path(file_path).is_file():
            self.errors.append(f"{file_type} path is not a file: {file_path}")
            self.log(f"Path is not a file", 'ERROR')
            return False
        
        self.log(f"{file_type} file exists", 'INFO')
        return True
    
    def validate_csv_format(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Validate CSV file can be read."""
        self.log(f"Reading {file_type} CSV file...")
        
        try:
            df = pd.read_csv(file_path)
            self.log(f"Successfully read {len(df)} rows", 'INFO')
            return df
        except pd.errors.EmptyDataError:
            self.errors.append(f"{file_type} file is empty")
            self.log(f"File is empty", 'ERROR')
            return None
        except pd.errors.ParserError as e:
            self.errors.append(f"{file_type} file has parsing errors: {e}")
            self.log(f"CSV parsing error: {e}", 'ERROR')
            return None
        except Exception as e:
            self.errors.append(f"Error reading {file_type} file: {e}")
            self.log(f"Error reading file: {e}", 'ERROR')
            return None
    
    def validate_columns(self, df: pd.DataFrame, required_columns: list,
                        optional_columns: list, file_type: str) -> bool:
        """Validate that required columns are present."""
        self.log(f"Validating {file_type} columns...")
        
        if df is None:
            return False
        
        actual_columns = set(df.columns)
        required_set = set(required_columns)
        optional_set = set(optional_columns) if optional_columns else set()
        
        # Check required columns
        missing_required = required_set - actual_columns
        if missing_required:
            self.errors.append(
                f"{file_type} missing required columns: {', '.join(missing_required)}"
            )
            self.log(f"Missing required columns: {', '.join(missing_required)}", 'ERROR')
            return False
        
        self.log(f"All required columns present", 'INFO')
        
        # Check optional columns
        missing_optional = optional_set - actual_columns
        if missing_optional:
            self.warnings.append(
                f"{file_type} missing optional columns: {', '.join(missing_optional)}"
            )
            self.log(f"Missing optional columns: {', '.join(missing_optional)}", 'WARNING')
        
        # Check for extra columns
        expected_columns = required_set | optional_set
        extra_columns = actual_columns - expected_columns
        if extra_columns:
            self.warnings.append(
                f"{file_type} has extra columns: {', '.join(extra_columns)}"
            )
            self.log(f"Extra columns found: {', '.join(extra_columns)}", 'WARNING')
        
        return True
    
    def validate_data_types(self, df: pd.DataFrame, file_type: str) -> bool:
        """Validate data types for numeric columns."""
        self.log(f"Validating {file_type} data types...")
        
        if df is None:
            return False
        
        has_errors = False
        
        # Check numeric columns for codes file
        if file_type == "Codes":
            numeric_columns = ['district_code', 'block_code', 'village_code']
            for col in numeric_columns:
                if col in df.columns:
                    try:
                        # Try to convert to numeric
                        pd.to_numeric(df[col], errors='raise')
                        self.log(f"Column '{col}' has valid numeric values", 'INFO')
                    except (ValueError, TypeError):
                        self.errors.append(
                            f"{file_type} column '{col}' contains non-numeric values"
                        )
                        self.log(f"Column '{col}' has non-numeric values", 'ERROR')
                        has_errors = True
        
        return not has_errors
    
    def validate_null_values(self, df: pd.DataFrame, required_columns: list,
                            file_type: str) -> bool:
        """Check for null values in required columns."""
        self.log(f"Checking {file_type} for null values...")
        
        if df is None:
            return False
        
        has_warnings = False
        
        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    null_pct = (null_count / len(df)) * 100
                    self.warnings.append(
                        f"{file_type} column '{col}' has {null_count} null values ({null_pct:.1f}%)"
                    )
                    self.log(
                        f"Column '{col}' has {null_count} null values ({null_pct:.1f}%)",
                        'WARNING'
                    )
                    has_warnings = True
        
        if not has_warnings:
            self.log("No null values in required columns", 'INFO')
        
        return True
    
    def validate_duplicates(self, df: pd.DataFrame, key_columns: list,
                           file_type: str) -> bool:
        """Check for duplicate records."""
        self.log(f"Checking {file_type} for duplicates...")
        
        if df is None:
            return False
        
        # Check if all key columns exist
        available_columns = [col for col in key_columns if col in df.columns]
        if not available_columns:
            return True
        
        duplicates = df.duplicated(subset=available_columns, keep=False)
        dup_count = duplicates.sum()
        
        if dup_count > 0:
            dup_pct = (dup_count / len(df)) * 100
            self.warnings.append(
                f"{file_type} has {dup_count} duplicate records ({dup_pct:.1f}%)"
            )
            self.log(
                f"Found {dup_count} duplicate records ({dup_pct:.1f}%)",
                'WARNING'
            )
        else:
            self.log("No duplicate records found", 'INFO')
        
        return True
    
    def validate_entities_file(self, file_path: str) -> bool:
        """Validate entities CSV file."""
        print("\n" + "="*60)
        print("VALIDATING ENTITIES FILE")
        print("="*60)
        
        if not self.validate_file_exists(file_path, "Entities"):
            return False
        
        df = self.validate_csv_format(file_path, "Entities")
        if df is None:
            return False
        
        # Validate required columns (district)
        if not self.validate_columns(df, self.ENTITIES_REQUIRED_COLUMNS, 
                                     self.ENTITIES_LOWER_LEVEL_COLUMNS, "Entities"):
            return False
        
        # Check that at least one lower level column is present
        actual_columns = set(df.columns)
        lower_level_present = any(col in actual_columns for col in self.ENTITIES_LOWER_LEVEL_COLUMNS)
        if not lower_level_present:
            self.errors.append(
                f"Entities file must contain at least one lower-level column: {', '.join(self.ENTITIES_LOWER_LEVEL_COLUMNS)}"
            )
            self.log(f"No lower-level columns found", 'ERROR')
            return False
        
        # Validate null values and duplicates using available columns
        all_columns = self.ENTITIES_REQUIRED_COLUMNS + [
            col for col in self.ENTITIES_LOWER_LEVEL_COLUMNS if col in df.columns
        ]
        self.validate_null_values(df, all_columns, "Entities")
        self.validate_duplicates(df, all_columns, "Entities")
        
        return True
    
    def validate_codes_file(self, file_path: str) -> bool:
        """Validate LGD codes CSV file."""
        print("\n" + "="*60)
        print("VALIDATING LGD CODES FILE")
        print("="*60)
        
        if not self.validate_file_exists(file_path, "Codes"):
            return False
        
        df = self.validate_csv_format(file_path, "Codes")
        if df is None:
            return False
        
        # Validate required columns (district_code and district)
        all_optional = self.CODES_OPTIONAL_COLUMNS.copy()
        for code_col, name_col in self.CODES_LOWER_LEVEL_PAIRS:
            all_optional.extend([code_col, name_col])
        
        if not self.validate_columns(df, self.CODES_REQUIRED_COLUMNS, all_optional, "Codes"):
            return False
        
        # Check that at least one lower level pair is present
        actual_columns = set(df.columns)
        has_lower_level = any(
            code_col in actual_columns and name_col in actual_columns
            for code_col, name_col in self.CODES_LOWER_LEVEL_PAIRS
        )
        if not has_lower_level:
            self.errors.append(
                f"Codes file must contain at least one lower-level pair (code + name): "
                f"{', '.join([f'{c}+{n}' for c, n in self.CODES_LOWER_LEVEL_PAIRS])}"
            )
            self.log(f"No lower-level pairs found", 'ERROR')
            return False
        
        if not self.validate_data_types(df, "Codes"):
            return False
        
        # Validate null values using available columns
        all_required = self.CODES_REQUIRED_COLUMNS.copy()
        for code_col, name_col in self.CODES_LOWER_LEVEL_PAIRS:
            if code_col in df.columns:
                all_required.append(code_col)
            if name_col in df.columns:
                all_required.append(name_col)
        
        self.validate_null_values(df, all_required, "Codes")
        
        # Check duplicates using available code columns
        dup_check_cols = [col for col in ['district_code', 'block_code', 'gp_code', 'village_code'] 
                         if col in df.columns]
        if dup_check_cols:
            self.validate_duplicates(df, dup_check_cols, "Codes")
        
        return True
    
    def print_summary(self) -> bool:
        """Print validation summary."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        if self.errors:
            print(f"\n✗ Found {len(self.errors)} error(s):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠ Found {len(self.warnings)} warning(s):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✓ All validations passed!")
            print("Your input files are ready for processing.")
            return True
        elif not self.errors:
            print("\n✓ No critical errors found.")
            print("Warnings indicate potential data quality issues but won't prevent processing.")
            return True
        else:
            print("\n✗ Validation failed!")
            print("Please fix the errors before running the mapping application.")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate input CSV files for LGD mapping application"
    )
    
    parser.add_argument(
        "--entities",
        required=True,
        help="Path to entities CSV file"
    )
    
    parser.add_argument(
        "--codes",
        required=True,
        help="Path to LGD codes CSV file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed validation output"
    )
    
    args = parser.parse_args()
    
    print("LGD Mapping Input Validation")
    print("="*60)
    
    validator = InputValidator(verbose=args.verbose)
    
    # Validate both files
    entities_valid = validator.validate_entities_file(args.entities)
    codes_valid = validator.validate_codes_file(args.codes)
    
    # Print summary
    success = validator.print_summary()
    
    # Exit with appropriate code
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
