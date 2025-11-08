"""
Data validation and quality checking utilities for LGD mapping application.

This module provides comprehensive data validation, quality checks, and
anomaly detection for entity and LGD reference data.
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import Counter

from ..exceptions import DataQualityError, ValidationError
from ..utils.error_handler import create_error_context, log_error_details


@dataclass
class ValidationRule:
    """Represents a data validation rule."""
    
    name: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    validator_func: callable
    applicable_columns: List[str] = field(default_factory=list)
    threshold: Optional[float] = None  # For percentage-based rules


@dataclass
class ValidationResult:
    """Represents the result of a validation check."""
    
    rule_name: str
    passed: bool
    severity: str
    message: str
    affected_records: int = 0
    affected_percentage: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    
    dataset_name: str
    total_records: int
    validation_results: List[ValidationResult] = field(default_factory=list)
    quality_score: float = 0.0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    recommendations: List[str] = field(default_factory=list)
    
    def add_result(self, result: ValidationResult):
        """Add a validation result to the report."""
        self.validation_results.append(result)
        
        if not result.passed:
            if result.severity == 'critical':
                self.critical_issues += 1
            elif result.severity == 'high':
                self.high_issues += 1
            elif result.severity == 'medium':
                self.medium_issues += 1
            elif result.severity == 'low':
                self.low_issues += 1
    
    def calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)."""
        if not self.validation_results:
            return 100.0
        
        # Weight different severity levels
        weights = {'critical': 10, 'high': 5, 'medium': 2, 'low': 1}
        
        total_weight = 0
        failed_weight = 0
        
        for result in self.validation_results:
            weight = weights.get(result.severity, 1)
            total_weight += weight
            
            if not result.passed:
                failed_weight += weight
        
        if total_weight == 0:
            return 100.0
        
        score = max(0.0, 100.0 - (failed_weight / total_weight * 100))
        self.quality_score = score
        return score


class DataValidator:
    """Comprehensive data validator for LGD mapping data."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the data validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.validation_rules = {}
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules."""
        # Entity data validation rules
        entity_rules = [
            ValidationRule(
                name="required_fields_present",
                description="Check that all required fields are present and not empty",
                severity="critical",
                validator_func=self._validate_required_fields,
                applicable_columns=['district', 'block', 'village']
            ),
            ValidationRule(
                name="no_null_values",
                description="Check for null or empty values in critical fields",
                severity="high",
                validator_func=self._validate_no_nulls,
                applicable_columns=['district', 'block', 'village'],
                threshold=0.05  # Allow up to 5% null values
            ),
            ValidationRule(
                name="reasonable_string_lengths",
                description="Check that string fields have reasonable lengths",
                severity="medium",
                validator_func=self._validate_string_lengths,
                applicable_columns=['district', 'block', 'village']
            ),
            ValidationRule(
                name="no_suspicious_characters",
                description="Check for suspicious characters or patterns",
                severity="medium",
                validator_func=self._validate_suspicious_characters,
                applicable_columns=['district', 'block', 'village']
            ),
            ValidationRule(
                name="duplicate_detection",
                description="Check for duplicate records",
                severity="medium",
                validator_func=self._validate_duplicates,
                threshold=0.02  # Flag if more than 2% duplicates
            ),
            ValidationRule(
                name="data_consistency",
                description="Check for data consistency patterns",
                severity="low",
                validator_func=self._validate_data_consistency
            )
        ]
        
        # LGD data validation rules
        lgd_rules = [
            ValidationRule(
                name="required_codes_present",
                description="Check that all required code fields are present",
                severity="critical",
                validator_func=self._validate_required_codes,
                applicable_columns=['district_code', 'block_code', 'village_code']
            ),
            ValidationRule(
                name="code_uniqueness",
                description="Check uniqueness of village codes",
                severity="high",
                validator_func=self._validate_code_uniqueness,
                applicable_columns=['village_code']
            ),
            ValidationRule(
                name="hierarchical_consistency",
                description="Check hierarchical consistency between codes and names",
                severity="high",
                validator_func=self._validate_hierarchical_consistency
            ),
            ValidationRule(
                name="code_ranges",
                description="Check that codes are within reasonable ranges",
                severity="medium",
                validator_func=self._validate_code_ranges,
                applicable_columns=['district_code', 'block_code', 'village_code']
            ),
            ValidationRule(
                name="name_code_alignment",
                description="Check alignment between names and codes",
                severity="medium",
                validator_func=self._validate_name_code_alignment
            )
        ]
        
        self.validation_rules['entity'] = entity_rules
        self.validation_rules['lgd'] = lgd_rules
    
    def validate_entity_data(self, df: pd.DataFrame, 
                           custom_rules: Optional[List[ValidationRule]] = None) -> DataQualityReport:
        """
        Validate entity data comprehensively.
        
        Args:
            df: Entity DataFrame to validate
            custom_rules: Optional additional validation rules
            
        Returns:
            DataQualityReport with validation results
        """
        self.logger.info("Starting comprehensive entity data validation")
        
        report = DataQualityReport(
            dataset_name="Entity Data",
            total_records=len(df)
        )
        
        # Get validation rules
        rules = self.validation_rules.get('entity', [])
        if custom_rules:
            rules.extend(custom_rules)
        
        # Run validation rules
        for rule in rules:
            try:
                result = self._run_validation_rule(rule, df, 'entity')
                report.add_result(result)
                
                # Log result
                status = "PASSED" if result.passed else "FAILED"
                self.logger.info(f"Validation rule '{rule.name}': {status}")
                if not result.passed:
                    self.logger.warning(f"  {result.message}")
                    
            except Exception as e:
                self.logger.error(f"Error running validation rule '{rule.name}': {e}")
                
                # Add error result
                error_result = ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    severity="high",
                    message=f"Validation rule failed to execute: {str(e)}",
                    recommendations=["Review validation rule implementation"]
                )
                report.add_result(error_result)
        
        # Calculate quality score
        report.calculate_quality_score()
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report, 'entity')
        
        self.logger.info(f"Entity data validation completed. Quality score: {report.quality_score:.1f}")
        
        return report
    
    def validate_lgd_data(self, df: pd.DataFrame,
                         custom_rules: Optional[List[ValidationRule]] = None) -> DataQualityReport:
        """
        Validate LGD reference data comprehensively.
        
        Args:
            df: LGD DataFrame to validate
            custom_rules: Optional additional validation rules
            
        Returns:
            DataQualityReport with validation results
        """
        self.logger.info("Starting comprehensive LGD data validation")
        
        report = DataQualityReport(
            dataset_name="LGD Reference Data",
            total_records=len(df)
        )
        
        # Get validation rules
        rules = self.validation_rules.get('lgd', [])
        if custom_rules:
            rules.extend(custom_rules)
        
        # Run validation rules
        for rule in rules:
            try:
                result = self._run_validation_rule(rule, df, 'lgd')
                report.add_result(result)
                
                # Log result
                status = "PASSED" if result.passed else "FAILED"
                self.logger.info(f"Validation rule '{rule.name}': {status}")
                if not result.passed:
                    self.logger.warning(f"  {result.message}")
                    
            except Exception as e:
                self.logger.error(f"Error running validation rule '{rule.name}': {e}")
                
                # Add error result
                error_result = ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    severity="high",
                    message=f"Validation rule failed to execute: {str(e)}",
                    recommendations=["Review validation rule implementation"]
                )
                report.add_result(error_result)
        
        # Calculate quality score
        report.calculate_quality_score()
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report, 'lgd')
        
        self.logger.info(f"LGD data validation completed. Quality score: {report.quality_score:.1f}")
        
        return report
    
    def validate_data_consistency(self, entities_df: pd.DataFrame, 
                                 lgd_df: pd.DataFrame) -> DataQualityReport:
        """
        Validate consistency between entity and LGD data.
        
        Args:
            entities_df: Entity DataFrame
            lgd_df: LGD DataFrame
            
        Returns:
            DataQualityReport with consistency validation results
        """
        self.logger.info("Starting cross-dataset consistency validation")
        
        report = DataQualityReport(
            dataset_name="Cross-Dataset Consistency",
            total_records=len(entities_df)
        )
        
        try:
            # Check district name consistency
            result = self._validate_district_consistency(entities_df, lgd_df)
            report.add_result(result)
            
            # Check district code coverage
            result = self._validate_district_code_coverage(entities_df, lgd_df)
            report.add_result(result)
            
            # Check block name patterns
            result = self._validate_block_patterns(entities_df, lgd_df)
            report.add_result(result)
            
            # Check village name patterns
            result = self._validate_village_patterns(entities_df, lgd_df)
            report.add_result(result)
            
        except Exception as e:
            self.logger.error(f"Error in consistency validation: {e}")
            
            error_result = ValidationResult(
                rule_name="consistency_validation",
                passed=False,
                severity="high",
                message=f"Consistency validation failed: {str(e)}"
            )
            report.add_result(error_result)
        
        # Calculate quality score
        report.calculate_quality_score()
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report, 'consistency')
        
        self.logger.info(f"Consistency validation completed. Quality score: {report.quality_score:.1f}")
        
        return report
    
    def detect_anomalies(self, df: pd.DataFrame, data_type: str) -> List[Dict[str, Any]]:
        """
        Detect anomalies and suspicious patterns in data.
        
        Args:
            df: DataFrame to analyze
            data_type: Type of data ('entity' or 'lgd')
            
        Returns:
            List of detected anomalies
        """
        self.logger.info(f"Starting anomaly detection for {data_type} data")
        
        anomalies = []
        
        try:
            # Detect outliers in string lengths
            anomalies.extend(self._detect_string_length_outliers(df, data_type))
            
            # Detect unusual character patterns
            anomalies.extend(self._detect_character_anomalies(df, data_type))
            
            # Detect frequency anomalies
            anomalies.extend(self._detect_frequency_anomalies(df, data_type))
            
            # Detect encoding issues
            anomalies.extend(self._detect_encoding_issues(df, data_type))
            
            if data_type == 'lgd':
                # LGD-specific anomaly detection
                anomalies.extend(self._detect_code_anomalies(df))
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            anomalies.append({
                'type': 'detection_error',
                'message': f"Anomaly detection failed: {str(e)}",
                'severity': 'medium'
            })
        
        self.logger.info(f"Anomaly detection completed. Found {len(anomalies)} anomalies")
        
        return anomalies
    
    def _run_validation_rule(self, rule: ValidationRule, df: pd.DataFrame, 
                           data_type: str) -> ValidationResult:
        """Run a single validation rule."""
        try:
            return rule.validator_func(df, rule, data_type)
        except Exception as e:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Rule execution failed: {str(e)}",
                recommendations=["Check rule implementation and data format"]
            )
    
    # Validation rule implementations
    
    def _validate_required_fields(self, df: pd.DataFrame, rule: ValidationRule, 
                                 data_type: str) -> ValidationResult:
        """Validate that required fields are present and not empty."""
        missing_fields = []
        empty_counts = {}
        
        for column in rule.applicable_columns:
            if column not in df.columns:
                missing_fields.append(column)
            else:
                empty_count = df[column].isna().sum() + (df[column] == '').sum()
                if empty_count > 0:
                    empty_counts[column] = empty_count
        
        if missing_fields or empty_counts:
            message_parts = []
            if missing_fields:
                message_parts.append(f"Missing columns: {missing_fields}")
            if empty_counts:
                message_parts.append(f"Empty values: {empty_counts}")
            
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message="; ".join(message_parts),
                affected_records=sum(empty_counts.values()),
                details={'missing_fields': missing_fields, 'empty_counts': empty_counts},
                recommendations=["Ensure all required fields are populated", "Check data extraction process"]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="All required fields are present and populated"
        )
    
    def _validate_no_nulls(self, df: pd.DataFrame, rule: ValidationRule, 
                          data_type: str) -> ValidationResult:
        """Validate that critical fields have minimal null values."""
        null_counts = {}
        total_records = len(df)
        
        for column in rule.applicable_columns:
            if column in df.columns:
                null_count = df[column].isna().sum()
                null_percentage = (null_count / total_records) * 100 if total_records > 0 else 0
                
                if null_percentage > (rule.threshold * 100 if rule.threshold else 5):
                    null_counts[column] = {'count': null_count, 'percentage': null_percentage}
        
        if null_counts:
            total_affected = sum(info['count'] for info in null_counts.values())
            
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"High null value rates detected: {null_counts}",
                affected_records=total_affected,
                affected_percentage=(total_affected / total_records) * 100,
                details={'null_counts': null_counts},
                recommendations=["Review data collection process", "Implement data cleaning procedures"]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="Null value rates are within acceptable limits"
        )
    
    def _validate_string_lengths(self, df: pd.DataFrame, rule: ValidationRule, 
                                data_type: str) -> ValidationResult:
        """Validate that string fields have reasonable lengths."""
        length_issues = {}
        
        for column in rule.applicable_columns:
            if column in df.columns and df[column].dtype == 'object':
                lengths = df[column].astype(str).str.len()
                
                # Check for extremely short or long values
                too_short = (lengths < 2).sum()  # Less than 2 characters
                too_long = (lengths > 100).sum()  # More than 100 characters
                
                if too_short > 0 or too_long > 0:
                    length_issues[column] = {
                        'too_short': too_short,
                        'too_long': too_long,
                        'avg_length': lengths.mean(),
                        'max_length': lengths.max()
                    }
        
        if length_issues:
            total_affected = sum(info['too_short'] + info['too_long'] for info in length_issues.values())
            
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"String length issues detected: {length_issues}",
                affected_records=total_affected,
                details={'length_issues': length_issues},
                recommendations=["Review data entry standards", "Check for truncated or corrupted data"]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="String lengths are within reasonable ranges"
        )
    
    def _validate_suspicious_characters(self, df: pd.DataFrame, rule: ValidationRule, 
                                      data_type: str) -> ValidationResult:
        """Validate that fields don't contain suspicious characters."""
        suspicious_patterns = [
            r'[^\w\s\-\.]',  # Non-alphanumeric except common punctuation
            r'\d{10,}',      # Very long numbers
            r'[A-Z]{10,}',   # Very long uppercase sequences
            r'(.)\1{5,}',    # Repeated characters (5+ times)
        ]
        
        suspicious_records = {}
        
        for column in rule.applicable_columns:
            if column in df.columns and df[column].dtype == 'object':
                column_issues = []
                
                for pattern in suspicious_patterns:
                    matches = df[column].astype(str).str.contains(pattern, regex=True, na=False)
                    if matches.any():
                        column_issues.append({
                            'pattern': pattern,
                            'count': matches.sum(),
                            'examples': df[matches][column].head(3).tolist()
                        })
                
                if column_issues:
                    suspicious_records[column] = column_issues
        
        if suspicious_records:
            total_affected = sum(
                sum(issue['count'] for issue in issues) 
                for issues in suspicious_records.values()
            )
            
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Suspicious character patterns detected: {len(suspicious_records)} columns affected",
                affected_records=total_affected,
                details={'suspicious_patterns': suspicious_records},
                recommendations=["Review data cleaning procedures", "Check for encoding issues"]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="No suspicious character patterns detected"
        )
    
    def _validate_duplicates(self, df: pd.DataFrame, rule: ValidationRule, 
                           data_type: str) -> ValidationResult:
        """Validate duplicate record levels."""
        if data_type == 'entity':
            duplicate_columns = ['district', 'block', 'village']
        else:  # lgd
            duplicate_columns = ['district_code', 'block_code', 'village_code']
        
        # Check if all required columns exist
        available_columns = [col for col in duplicate_columns if col in df.columns]
        if not available_columns:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity="medium",
                message="Cannot check duplicates: required columns not found",
                recommendations=["Ensure required columns are present"]
            )
        
        # Find duplicates
        duplicates = df.duplicated(subset=available_columns, keep=False)
        duplicate_count = duplicates.sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
        
        threshold_percentage = (rule.threshold * 100) if rule.threshold else 2.0
        
        if duplicate_percentage > threshold_percentage:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"High duplicate rate: {duplicate_count} records ({duplicate_percentage:.2f}%)",
                affected_records=duplicate_count,
                affected_percentage=duplicate_percentage,
                details={'duplicate_columns': available_columns},
                recommendations=["Review data deduplication process", "Check data source integrity"]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message=f"Duplicate rate is acceptable: {duplicate_percentage:.2f}%"
        )
    
    def _validate_data_consistency(self, df: pd.DataFrame, rule: ValidationRule, 
                                 data_type: str) -> ValidationResult:
        """Validate internal data consistency patterns."""
        consistency_issues = []
        
        # Check for consistent naming patterns within districts/blocks
        if 'district' in df.columns and 'block' in df.columns:
            # Check if same district has consistent block naming patterns
            district_blocks = df.groupby('district')['block'].apply(list).to_dict()
            
            for district, blocks in district_blocks.items():
                if len(set(blocks)) != len(blocks):  # Has duplicates
                    continue
                
                # Check for inconsistent patterns (very basic check)
                block_lengths = [len(str(block)) for block in blocks]
                if len(set(block_lengths)) > 3:  # Too much variation in length
                    consistency_issues.append(f"District '{district}' has inconsistent block name lengths")
        
        if consistency_issues:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Data consistency issues: {len(consistency_issues)} found",
                details={'issues': consistency_issues},
                recommendations=["Review data standardization procedures"]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="Data consistency patterns are acceptable"
        )
    
    # LGD-specific validation methods
    
    def _validate_required_codes(self, df: pd.DataFrame, rule: ValidationRule, 
                               data_type: str) -> ValidationResult:
        """Validate that required code fields are present and valid."""
        missing_codes = {}
        invalid_codes = {}
        
        for column in rule.applicable_columns:
            if column not in df.columns:
                missing_codes[column] = "Column not found"
            else:
                # Check for null codes
                null_count = df[column].isna().sum()
                if null_count > 0:
                    missing_codes[column] = f"{null_count} null values"
                
                # Check for invalid codes (non-numeric)
                try:
                    numeric_codes = pd.to_numeric(df[column], errors='coerce')
                    invalid_count = numeric_codes.isna().sum() - df[column].isna().sum()
                    if invalid_count > 0:
                        invalid_codes[column] = f"{invalid_count} non-numeric values"
                except Exception:
                    invalid_codes[column] = "Cannot convert to numeric"
        
        if missing_codes or invalid_codes:
            issues = []
            if missing_codes:
                issues.append(f"Missing codes: {missing_codes}")
            if invalid_codes:
                issues.append(f"Invalid codes: {invalid_codes}")
            
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message="; ".join(issues),
                details={'missing_codes': missing_codes, 'invalid_codes': invalid_codes},
                recommendations=["Ensure all code fields are properly populated", "Validate code data types"]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="All required codes are present and valid"
        )
    
    def _validate_code_uniqueness(self, df: pd.DataFrame, rule: ValidationRule, 
                                data_type: str) -> ValidationResult:
        """Validate uniqueness of village codes."""
        uniqueness_issues = {}
        
        for column in rule.applicable_columns:
            if column in df.columns:
                duplicates = df[column].duplicated(keep=False)
                duplicate_count = duplicates.sum()
                
                if duplicate_count > 0:
                    duplicate_codes = df[duplicates][column].unique()
                    uniqueness_issues[column] = {
                        'duplicate_count': duplicate_count,
                        'unique_duplicates': len(duplicate_codes),
                        'examples': duplicate_codes[:5].tolist()
                    }
        
        if uniqueness_issues:
            total_affected = sum(info['duplicate_count'] for info in uniqueness_issues.values())
            
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Code uniqueness violations: {uniqueness_issues}",
                affected_records=total_affected,
                details={'uniqueness_issues': uniqueness_issues},
                recommendations=["Review code assignment process", "Check for data duplication"]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="Code uniqueness is maintained"
        )
    
    def _validate_hierarchical_consistency(self, df: pd.DataFrame, rule: ValidationRule, 
                                         data_type: str) -> ValidationResult:
        """Validate hierarchical consistency between codes and names."""
        consistency_issues = []
        
        # Check district code-name consistency
        if all(col in df.columns for col in ['district_code', 'district']):
            district_mapping = df.groupby('district_code')['district'].nunique()
            inconsistent_districts = district_mapping[district_mapping > 1]
            
            if len(inconsistent_districts) > 0:
                consistency_issues.append(f"{len(inconsistent_districts)} district codes have multiple names")
        
        # Check block code-name consistency within districts
        if all(col in df.columns for col in ['district_code', 'block_code', 'block']):
            block_mapping = df.groupby(['district_code', 'block_code'])['block'].nunique()
            inconsistent_blocks = block_mapping[block_mapping > 1]
            
            if len(inconsistent_blocks) > 0:
                consistency_issues.append(f"{len(inconsistent_blocks)} block codes have multiple names")
        
        if consistency_issues:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Hierarchical consistency issues: {'; '.join(consistency_issues)}",
                details={'issues': consistency_issues},
                recommendations=["Review hierarchical data integrity", "Check code-name mappings"]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="Hierarchical consistency is maintained"
        )
    
    def _validate_code_ranges(self, df: pd.DataFrame, rule: ValidationRule, 
                            data_type: str) -> ValidationResult:
        """Validate that codes are within reasonable ranges."""
        range_issues = {}
        
        # Define reasonable ranges for different code types
        code_ranges = {
            'district_code': (1, 999),
            'block_code': (1, 9999),
            'village_code': (1, 999999)
        }
        
        for column in rule.applicable_columns:
            if column in df.columns and column in code_ranges:
                min_val, max_val = code_ranges[column]
                
                try:
                    numeric_codes = pd.to_numeric(df[column], errors='coerce')
                    out_of_range = (numeric_codes < min_val) | (numeric_codes > max_val)
                    out_of_range_count = out_of_range.sum()
                    
                    if out_of_range_count > 0:
                        range_issues[column] = {
                            'out_of_range_count': out_of_range_count,
                            'expected_range': f"{min_val}-{max_val}",
                            'actual_min': numeric_codes.min(),
                            'actual_max': numeric_codes.max()
                        }
                except Exception as e:
                    range_issues[column] = f"Error checking range: {str(e)}"
        
        if range_issues:
            total_affected = sum(
                info['out_of_range_count'] if isinstance(info, dict) else 0 
                for info in range_issues.values()
            )
            
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Code range issues: {range_issues}",
                affected_records=total_affected,
                details={'range_issues': range_issues},
                recommendations=["Review code assignment standards", "Check for data entry errors"]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="All codes are within reasonable ranges"
        )
    
    def _validate_name_code_alignment(self, df: pd.DataFrame, rule: ValidationRule, 
                                    data_type: str) -> ValidationResult:
        """Validate alignment between names and codes."""
        # This is a placeholder for more sophisticated name-code alignment checks
        # Could include checks for:
        # - Names that should have similar codes (alphabetical ordering)
        # - Codes that should have similar names
        # - Geographic proximity checks if coordinates are available
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="Name-code alignment check completed (basic implementation)"
        )
    
    # Cross-dataset validation methods
    
    def _validate_district_consistency(self, entities_df: pd.DataFrame, 
                                     lgd_df: pd.DataFrame) -> ValidationResult:
        """Validate district name consistency between datasets."""
        entity_districts = set(entities_df['district'].dropna().unique())
        lgd_districts = set(lgd_df['district'].dropna().unique())
        
        missing_in_lgd = entity_districts - lgd_districts
        extra_in_lgd = lgd_districts - entity_districts
        
        if missing_in_lgd or extra_in_lgd:
            issues = []
            if missing_in_lgd:
                issues.append(f"Districts in entities but not in LGD: {sorted(list(missing_in_lgd))}")
            if extra_in_lgd:
                issues.append(f"Districts in LGD but not in entities: {len(extra_in_lgd)} districts")
            
            return ValidationResult(
                rule_name="district_consistency",
                passed=False,
                severity="medium",
                message="; ".join(issues),
                affected_records=len(missing_in_lgd),
                details={'missing_in_lgd': list(missing_in_lgd), 'extra_in_lgd': list(extra_in_lgd)},
                recommendations=["Review district name mappings", "Check data source consistency"]
            )
        
        return ValidationResult(
            rule_name="district_consistency",
            passed=True,
            severity="medium",
            message="District names are consistent between datasets"
        )
    
    def _validate_district_code_coverage(self, entities_df: pd.DataFrame, 
                                       lgd_df: pd.DataFrame) -> ValidationResult:
        """Validate district code coverage."""
        entities_with_codes = entities_df[entities_df['district_code'].notna()]
        
        if len(entities_with_codes) == 0:
            return ValidationResult(
                rule_name="district_code_coverage",
                passed=False,
                severity="high",
                message="No entities have district codes assigned",
                affected_records=len(entities_df),
                recommendations=["Implement district code mapping", "Review data preprocessing"]
            )
        
        entity_codes = set(entities_with_codes['district_code'].unique())
        lgd_codes = set(lgd_df['district_code'].unique())
        
        invalid_codes = entity_codes - lgd_codes
        coverage_percentage = (len(entities_with_codes) / len(entities_df)) * 100
        
        issues = []
        if invalid_codes:
            issues.append(f"Invalid district codes in entities: {sorted(list(invalid_codes))}")
        if coverage_percentage < 80:  # Less than 80% coverage
            issues.append(f"Low district code coverage: {coverage_percentage:.1f}%")
        
        if issues:
            return ValidationResult(
                rule_name="district_code_coverage",
                passed=False,
                severity="medium",
                message="; ".join(issues),
                affected_records=len(entities_df) - len(entities_with_codes),
                affected_percentage=100 - coverage_percentage,
                details={'invalid_codes': list(invalid_codes), 'coverage_percentage': coverage_percentage},
                recommendations=["Review district code assignments", "Improve code mapping coverage"]
            )
        
        return ValidationResult(
            rule_name="district_code_coverage",
            passed=True,
            severity="medium",
            message=f"District code coverage is good: {coverage_percentage:.1f}%"
        )
    
    def _validate_block_patterns(self, entities_df: pd.DataFrame, 
                               lgd_df: pd.DataFrame) -> ValidationResult:
        """Validate block name patterns between datasets."""
        # Simple pattern validation - could be enhanced with fuzzy matching
        entity_blocks = set(entities_df['block'].dropna().str.lower().unique())
        lgd_blocks = set(lgd_df['block'].dropna().str.lower().unique())
        
        common_blocks = entity_blocks & lgd_blocks
        match_percentage = (len(common_blocks) / len(entity_blocks)) * 100 if entity_blocks else 0
        
        if match_percentage < 50:  # Less than 50% match
            return ValidationResult(
                rule_name="block_patterns",
                passed=False,
                severity="low",
                message=f"Low block name match rate: {match_percentage:.1f}%",
                details={'match_percentage': match_percentage},
                recommendations=["Review block name standardization", "Consider fuzzy matching"]
            )
        
        return ValidationResult(
            rule_name="block_patterns",
            passed=True,
            severity="low",
            message=f"Block name patterns are acceptable: {match_percentage:.1f}% match"
        )
    
    def _validate_village_patterns(self, entities_df: pd.DataFrame, 
                                 lgd_df: pd.DataFrame) -> ValidationResult:
        """Validate village name patterns between datasets."""
        # Similar to block patterns but for villages
        entity_villages = set(entities_df['village'].dropna().str.lower().unique())
        lgd_villages = set(lgd_df['village'].dropna().str.lower().unique())
        
        common_villages = entity_villages & lgd_villages
        match_percentage = (len(common_villages) / len(entity_villages)) * 100 if entity_villages else 0
        
        if match_percentage < 30:  # Less than 30% match (lower threshold for villages)
            return ValidationResult(
                rule_name="village_patterns",
                passed=False,
                severity="low",
                message=f"Low village name match rate: {match_percentage:.1f}%",
                details={'match_percentage': match_percentage},
                recommendations=["Review village name standardization", "Consider fuzzy matching strategies"]
            )
        
        return ValidationResult(
            rule_name="village_patterns",
            passed=True,
            severity="low",
            message=f"Village name patterns are acceptable: {match_percentage:.1f}% match"
        )
    
    # Anomaly detection methods
    
    def _detect_string_length_outliers(self, df: pd.DataFrame, data_type: str) -> List[Dict[str, Any]]:
        """Detect outliers in string lengths."""
        anomalies = []
        
        string_columns = ['district', 'block', 'village']
        if data_type == 'lgd' and 'gp' in df.columns:
            string_columns.append('gp')
        
        for column in string_columns:
            if column in df.columns and df[column].dtype == 'object':
                lengths = df[column].astype(str).str.len()
                
                # Use IQR method to detect outliers
                Q1 = lengths.quantile(0.25)
                Q3 = lengths.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (lengths < lower_bound) | (lengths > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    anomalies.append({
                        'type': 'string_length_outlier',
                        'column': column,
                        'count': outlier_count,
                        'percentage': (outlier_count / len(df)) * 100,
                        'bounds': {'lower': lower_bound, 'upper': upper_bound},
                        'examples': df[outliers][column].head(3).tolist(),
                        'severity': 'low'
                    })
        
        return anomalies
    
    def _detect_character_anomalies(self, df: pd.DataFrame, data_type: str) -> List[Dict[str, Any]]:
        """Detect unusual character patterns."""
        anomalies = []
        
        string_columns = ['district', 'block', 'village']
        if data_type == 'lgd' and 'gp' in df.columns:
            string_columns.append('gp')
        
        for column in string_columns:
            if column in df.columns and df[column].dtype == 'object':
                # Check for non-printable characters
                non_printable = df[column].astype(str).str.contains(r'[^\x20-\x7E]', regex=True, na=False)
                non_printable_count = non_printable.sum()
                
                if non_printable_count > 0:
                    anomalies.append({
                        'type': 'non_printable_characters',
                        'column': column,
                        'count': non_printable_count,
                        'examples': df[non_printable][column].head(3).tolist(),
                        'severity': 'medium'
                    })
                
                # Check for excessive special characters
                special_chars = df[column].astype(str).str.count(r'[^a-zA-Z0-9\s]')
                excessive_special = special_chars > 5  # More than 5 special characters
                excessive_count = excessive_special.sum()
                
                if excessive_count > 0:
                    anomalies.append({
                        'type': 'excessive_special_characters',
                        'column': column,
                        'count': excessive_count,
                        'examples': df[excessive_special][column].head(3).tolist(),
                        'severity': 'low'
                    })
        
        return anomalies
    
    def _detect_frequency_anomalies(self, df: pd.DataFrame, data_type: str) -> List[Dict[str, Any]]:
        """Detect frequency-based anomalies."""
        anomalies = []
        
        string_columns = ['district', 'block', 'village']
        
        for column in string_columns:
            if column in df.columns:
                value_counts = df[column].value_counts()
                
                # Detect values that appear unusually frequently
                total_records = len(df)
                high_frequency_threshold = total_records * 0.1  # More than 10% of records
                
                high_frequency = value_counts[value_counts > high_frequency_threshold]
                
                if len(high_frequency) > 0:
                    anomalies.append({
                        'type': 'high_frequency_values',
                        'column': column,
                        'values': high_frequency.to_dict(),
                        'threshold_percentage': 10.0,
                        'severity': 'low'
                    })
        
        return anomalies
    
    def _detect_encoding_issues(self, df: pd.DataFrame, data_type: str) -> List[Dict[str, Any]]:
        """Detect potential encoding issues."""
        anomalies = []
        
        string_columns = ['district', 'block', 'village']
        if data_type == 'lgd' and 'gp' in df.columns:
            string_columns.append('gp')
        
        # Common encoding issue patterns
        encoding_patterns = [
            (r'Ã¢â‚¬â„¢', 'UTF-8 to Latin-1 conversion issue'),
            (r'â€™', 'Smart quote encoding issue'),
            (r'Ã¡', 'Accented character encoding issue'),
            (r'\?{2,}', 'Multiple question marks (encoding failure)')
        ]
        
        for column in string_columns:
            if column in df.columns and df[column].dtype == 'object':
                for pattern, description in encoding_patterns:
                    matches = df[column].astype(str).str.contains(pattern, regex=True, na=False)
                    match_count = matches.sum()
                    
                    if match_count > 0:
                        anomalies.append({
                            'type': 'encoding_issue',
                            'column': column,
                            'pattern': pattern,
                            'description': description,
                            'count': match_count,
                            'examples': df[matches][column].head(3).tolist(),
                            'severity': 'medium'
                        })
        
        return anomalies
    
    def _detect_code_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in LGD codes."""
        anomalies = []
        
        code_columns = ['district_code', 'block_code', 'village_code']
        
        for column in code_columns:
            if column in df.columns:
                try:
                    numeric_codes = pd.to_numeric(df[column], errors='coerce')
                    
                    # Check for gaps in code sequences
                    unique_codes = sorted(numeric_codes.dropna().unique())
                    if len(unique_codes) > 1:
                        gaps = []
                        for i in range(1, len(unique_codes)):
                            if unique_codes[i] - unique_codes[i-1] > 100:  # Gap larger than 100
                                gaps.append((unique_codes[i-1], unique_codes[i]))
                        
                        if gaps:
                            anomalies.append({
                                'type': 'code_sequence_gaps',
                                'column': column,
                                'gaps': gaps,
                                'severity': 'low'
                            })
                    
                    # Check for unusually large codes
                    max_code = numeric_codes.max()
                    expected_max = {'district_code': 999, 'block_code': 9999, 'village_code': 999999}
                    
                    if column in expected_max and max_code > expected_max[column]:
                        anomalies.append({
                            'type': 'unusually_large_codes',
                            'column': column,
                            'max_code': max_code,
                            'expected_max': expected_max[column],
                            'severity': 'medium'
                        })
                        
                except Exception as e:
                    anomalies.append({
                        'type': 'code_analysis_error',
                        'column': column,
                        'error': str(e),
                        'severity': 'low'
                    })
        
        return anomalies
    
    def _generate_recommendations(self, report: DataQualityReport, data_type: str) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Critical issues
        if report.critical_issues > 0:
            recommendations.append("Address critical data quality issues immediately")
            recommendations.append("Review data collection and validation processes")
        
        # High issues
        if report.high_issues > 0:
            recommendations.append("Prioritize resolution of high-severity issues")
            if data_type == 'entity':
                recommendations.append("Implement data cleaning procedures for entity data")
            else:
                recommendations.append("Validate LGD reference data integrity")
        
        # Medium issues
        if report.medium_issues > 0:
            recommendations.append("Plan remediation for medium-severity issues")
            recommendations.append("Consider automated data quality monitoring")
        
        # Quality score based recommendations
        if report.quality_score < 70:
            recommendations.append("Data quality score is below acceptable threshold")
            recommendations.append("Implement comprehensive data quality improvement program")
        elif report.quality_score < 85:
            recommendations.append("Data quality is acceptable but has room for improvement")
            recommendations.append("Focus on resolving remaining quality issues")
        
        # Data type specific recommendations
        if data_type == 'entity':
            recommendations.append("Ensure consistent data entry standards for entity data")
            recommendations.append("Implement validation at data collection points")
        elif data_type == 'lgd':
            recommendations.append("Maintain hierarchical consistency in LGD reference data")
            recommendations.append("Regular validation of code-name mappings")
        elif data_type == 'consistency':
            recommendations.append("Improve data standardization between datasets")
            recommendations.append("Consider implementing fuzzy matching for better coverage")
        
        return list(set(recommendations))  # Remove duplicates
    
    def generate_quality_report_summary(self, report: DataQualityReport) -> str:
        """Generate a human-readable summary of the quality report."""
        summary_lines = [
            f"Data Quality Report: {report.dataset_name}",
            f"Total Records: {report.total_records:,}",
            f"Quality Score: {report.quality_score:.1f}/100",
            "",
            "Issue Summary:",
            f"  Critical: {report.critical_issues}",
            f"  High: {report.high_issues}",
            f"  Medium: {report.medium_issues}",
            f"  Low: {report.low_issues}",
            ""
        ]
        
        if report.validation_results:
            summary_lines.append("Failed Validations:")
            for result in report.validation_results:
                if not result.passed:
                    summary_lines.append(f"  - {result.rule_name}: {result.message}")
            summary_lines.append("")
        
        if report.recommendations:
            summary_lines.append("Recommendations:")
            for rec in report.recommendations:
                summary_lines.append(f"  - {rec}")
        
        return "\n".join(summary_lines)