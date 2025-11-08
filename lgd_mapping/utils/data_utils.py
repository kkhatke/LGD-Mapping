"""
Data utility functions for type conversions and null handling.

This module provides utility functions for cleaning and converting data types,
handling null values, and validating data integrity.
"""

import pandas as pd
from typing import Any, Optional, Union
import numpy as np


def safe_int_conversion(value: Any) -> Optional[int]:
    """
    Safely convert a value to integer, handling nulls and invalid values.
    
    Args:
        value: Value to convert to integer
        
    Returns:
        Integer value or None if conversion fails
    """
    if pd.isna(value) or value is None or value == '':
        return None
    
    try:
        # Handle string representations of floats (e.g., "123.0")
        if isinstance(value, str):
            value = value.strip()
            if '.' in value:
                value = float(value)
        
        return int(float(value))
    except (ValueError, TypeError):
        return None


def safe_string_conversion(value: Any) -> str:
    """
    Safely convert a value to string, handling nulls and cleaning whitespace.
    
    Args:
        value: Value to convert to string
        
    Returns:
        Cleaned string value or empty string if null
    """
    if pd.isna(value) or value is None:
        return ""
    
    return str(value).strip()


def normalize_string(value: str) -> str:
    """
    Normalize string by removing extra whitespace and converting to title case.
    
    Args:
        value: String to normalize
        
    Returns:
        Normalized string
    """
    if not value:
        return ""
    
    # Remove extra whitespace and normalize case
    normalized = ' '.join(value.strip().split())
    return normalized.title()


def is_null_or_empty(value: Any) -> bool:
    """
    Check if a value is null, empty, or contains only whitespace.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is null/empty, False otherwise
    """
    if pd.isna(value) or value is None:
        return True
    
    if isinstance(value, str):
        return not value.strip()
    
    return False


def validate_required_fields(record: dict, required_fields: list) -> tuple[bool, list]:
    """
    Validate that all required fields are present and not empty.
    
    Args:
        record: Dictionary containing record data
        required_fields: List of field names that are required
        
    Returns:
        Tuple of (is_valid, list_of_missing_fields)
    """
    missing_fields = []
    
    for field in required_fields:
        if field not in record or is_null_or_empty(record[field]):
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields


def clean_dataframe_strings(df: pd.DataFrame, string_columns: list) -> pd.DataFrame:
    """
    Clean string columns in a DataFrame by removing extra whitespace.
    
    Args:
        df: DataFrame to clean
        string_columns: List of column names to clean
        
    Returns:
        DataFrame with cleaned string columns
    """
    df_cleaned = df.copy()
    
    for col in string_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(safe_string_conversion)
    
    return df_cleaned


def convert_numeric_columns(df: pd.DataFrame, numeric_columns: dict) -> pd.DataFrame:
    """
    Convert specified columns to numeric types with error handling.
    
    Args:
        df: DataFrame to process
        numeric_columns: Dict mapping column names to target types ('int' or 'float')
        
    Returns:
        DataFrame with converted numeric columns
    """
    df_converted = df.copy()
    
    for col, target_type in numeric_columns.items():
        if col in df_converted.columns:
            if target_type == 'int':
                df_converted[col] = df_converted[col].apply(safe_int_conversion)
            elif target_type == 'float':
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
    
    return df_converted


def detect_duplicates(df: pd.DataFrame, key_columns: list) -> pd.DataFrame:
    """
    Detect duplicate records based on specified key columns.
    
    Args:
        df: DataFrame to check for duplicates
        key_columns: List of column names to use for duplicate detection
        
    Returns:
        DataFrame containing only the duplicate records
    """
    # Create a subset with only the key columns for duplicate detection
    subset_df = df[key_columns].copy()
    
    # Find duplicates
    duplicated_mask = subset_df.duplicated(keep=False)
    
    return df[duplicated_mask].copy()


def get_data_quality_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of data quality metrics for a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing data quality metrics
    """
    summary = {
        'total_records': len(df),
        'null_counts': df.isnull().sum().to_dict(),
        'empty_string_counts': {},
        'duplicate_count': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Count empty strings in string columns
    for col in df.select_dtypes(include=['object']).columns:
        empty_count = (df[col].astype(str).str.strip() == '').sum()
        summary['empty_string_counts'][col] = empty_count
    
    return summary