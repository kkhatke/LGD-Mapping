"""
Custom exception classes for LGD mapping application.

This module defines custom exception classes for different types of errors
that can occur during the mapping process, providing specific error handling
and recovery mechanisms.
"""

from typing import Optional, List, Dict, Any


class LGDMappingError(Exception):
    """Base exception class for all LGD mapping errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize the base mapping error.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            context: Optional context information about the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context
        }


class ValidationError(LGDMappingError):
    """Exception raised for data validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 invalid_value: Any = None, validation_rules: Optional[List[str]] = None):
        """
        Initialize validation error.
        
        Args:
            message: Human-readable error message
            field_name: Name of the field that failed validation
            invalid_value: The invalid value that caused the error
            validation_rules: List of validation rules that were violated
        """
        context = {
            'field_name': field_name,
            'invalid_value': str(invalid_value) if invalid_value is not None else None,
            'validation_rules': validation_rules or []
        }
        super().__init__(message, error_code='VALIDATION_ERROR', context=context)
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.validation_rules = validation_rules or []


class DataLoadError(LGDMappingError):
    """Exception raised for data loading errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 line_number: Optional[int] = None, original_error: Optional[Exception] = None):
        """
        Initialize data load error.
        
        Args:
            message: Human-readable error message
            file_path: Path to the file that caused the error
            line_number: Line number where the error occurred
            original_error: Original exception that caused this error
        """
        context = {
            'file_path': file_path,
            'line_number': line_number,
            'original_error': str(original_error) if original_error else None,
            'original_error_type': type(original_error).__name__ if original_error else None
        }
        super().__init__(message, error_code='DATA_LOAD_ERROR', context=context)
        self.file_path = file_path
        self.line_number = line_number
        self.original_error = original_error


class FileAccessError(LGDMappingError):
    """Exception raised for file access and I/O errors."""
    
    def __init__(self, message: str, file_path: str, operation: str, 
                 original_error: Optional[Exception] = None):
        """
        Initialize file access error.
        
        Args:
            message: Human-readable error message
            file_path: Path to the file that caused the error
            operation: Type of operation that failed (read, write, create, etc.)
            original_error: Original exception that caused this error
        """
        context = {
            'file_path': file_path,
            'operation': operation,
            'original_error': str(original_error) if original_error else None,
            'original_error_type': type(original_error).__name__ if original_error else None
        }
        super().__init__(message, error_code='FILE_ACCESS_ERROR', context=context)
        self.file_path = file_path
        self.operation = operation
        self.original_error = original_error


class MappingProcessError(LGDMappingError):
    """Exception raised for errors during the mapping process."""
    
    def __init__(self, message: str, strategy: Optional[str] = None, 
                 entity_count: Optional[int] = None, processed_count: Optional[int] = None,
                 original_error: Optional[Exception] = None):
        """
        Initialize mapping process error.
        
        Args:
            message: Human-readable error message
            strategy: Name of the matching strategy that failed
            entity_count: Total number of entities being processed
            processed_count: Number of entities processed before failure
            original_error: Original exception that caused this error
        """
        context = {
            'strategy': strategy,
            'entity_count': entity_count,
            'processed_count': processed_count,
            'original_error': str(original_error) if original_error else None,
            'original_error_type': type(original_error).__name__ if original_error else None
        }
        super().__init__(message, error_code='MAPPING_PROCESS_ERROR', context=context)
        self.strategy = strategy
        self.entity_count = entity_count
        self.processed_count = processed_count
        self.original_error = original_error


class ConfigurationError(LGDMappingError):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 config_value: Any = None, valid_values: Optional[List[Any]] = None):
        """
        Initialize configuration error.
        
        Args:
            message: Human-readable error message
            config_key: Configuration key that has invalid value
            config_value: Invalid configuration value
            valid_values: List of valid values for the configuration key
        """
        context = {
            'config_key': config_key,
            'config_value': str(config_value) if config_value is not None else None,
            'valid_values': [str(v) for v in valid_values] if valid_values else None
        }
        super().__init__(message, error_code='CONFIGURATION_ERROR', context=context)
        self.config_key = config_key
        self.config_value = config_value
        self.valid_values = valid_values or []


class MatchingError(LGDMappingError):
    """Exception raised for errors during matching operations."""
    
    def __init__(self, message: str, entity_data: Optional[Dict[str, Any]] = None,
                 matcher_type: Optional[str] = None, threshold: Optional[float] = None,
                 original_error: Optional[Exception] = None):
        """
        Initialize matching error.
        
        Args:
            message: Human-readable error message
            entity_data: Data of the entity that caused the error
            matcher_type: Type of matcher that failed (exact, fuzzy, etc.)
            threshold: Matching threshold if applicable
            original_error: Original exception that caused this error
        """
        context = {
            'entity_data': entity_data,
            'matcher_type': matcher_type,
            'threshold': threshold,
            'original_error': str(original_error) if original_error else None,
            'original_error_type': type(original_error).__name__ if original_error else None
        }
        super().__init__(message, error_code='MATCHING_ERROR', context=context)
        self.entity_data = entity_data or {}
        self.matcher_type = matcher_type
        self.threshold = threshold
        self.original_error = original_error


class OutputGenerationError(LGDMappingError):
    """Exception raised for errors during output generation."""
    
    def __init__(self, message: str, output_type: Optional[str] = None,
                 output_path: Optional[str] = None, record_count: Optional[int] = None,
                 original_error: Optional[Exception] = None):
        """
        Initialize output generation error.
        
        Args:
            message: Human-readable error message
            output_type: Type of output being generated (CSV, report, etc.)
            output_path: Path where output was being written
            record_count: Number of records being written
            original_error: Original exception that caused this error
        """
        context = {
            'output_type': output_type,
            'output_path': output_path,
            'record_count': record_count,
            'original_error': str(original_error) if original_error else None,
            'original_error_type': type(original_error).__name__ if original_error else None
        }
        super().__init__(message, error_code='OUTPUT_GENERATION_ERROR', context=context)
        self.output_type = output_type
        self.output_path = output_path
        self.record_count = record_count
        self.original_error = original_error


class DataQualityError(LGDMappingError):
    """Exception raised for data quality issues."""
    
    def __init__(self, message: str, quality_issue: str, affected_records: Optional[int] = None,
                 severity: str = 'medium', recommendations: Optional[List[str]] = None):
        """
        Initialize data quality error.
        
        Args:
            message: Human-readable error message
            quality_issue: Type of quality issue (duplicates, missing_data, etc.)
            affected_records: Number of records affected by the issue
            severity: Severity level (low, medium, high, critical)
            recommendations: List of recommendations to fix the issue
        """
        context = {
            'quality_issue': quality_issue,
            'affected_records': affected_records,
            'severity': severity,
            'recommendations': recommendations or []
        }
        super().__init__(message, error_code='DATA_QUALITY_ERROR', context=context)
        self.quality_issue = quality_issue
        self.affected_records = affected_records
        self.severity = severity
        self.recommendations = recommendations or []


class RecoveryError(LGDMappingError):
    """Exception raised when error recovery mechanisms fail."""
    
    def __init__(self, message: str, recovery_strategy: str, 
                 original_error: Exception, recovery_attempts: int = 1):
        """
        Initialize recovery error.
        
        Args:
            message: Human-readable error message
            recovery_strategy: Name of the recovery strategy that failed
            original_error: Original error that triggered recovery
            recovery_attempts: Number of recovery attempts made
        """
        context = {
            'recovery_strategy': recovery_strategy,
            'original_error': str(original_error),
            'original_error_type': type(original_error).__name__,
            'recovery_attempts': recovery_attempts
        }
        super().__init__(message, error_code='RECOVERY_ERROR', context=context)
        self.recovery_strategy = recovery_strategy
        self.original_error = original_error
        self.recovery_attempts = recovery_attempts


class HierarchyDetectionError(MappingProcessError):
    """Exception raised when hierarchy detection fails."""
    
    def __init__(self, message: str, detected_levels: Optional[List[str]] = None,
                 missing_required_levels: Optional[List[str]] = None,
                 inconsistencies: Optional[List[str]] = None,
                 original_error: Optional[Exception] = None):
        """
        Initialize hierarchy detection error.
        
        Args:
            message: Human-readable error message
            detected_levels: List of hierarchical levels that were detected
            missing_required_levels: List of required levels that are missing
            inconsistencies: List of hierarchical inconsistencies found
            original_error: Original exception that caused this error
        """
        context = {
            'detected_levels': detected_levels or [],
            'missing_required_levels': missing_required_levels or [],
            'inconsistencies': inconsistencies or [],
            'original_error': str(original_error) if original_error else None,
            'original_error_type': type(original_error).__name__ if original_error else None
        }
        super().__init__(
            message=message,
            strategy='hierarchy_detection',
            original_error=original_error
        )
        self.context.update(context)
        self.detected_levels = detected_levels or []
        self.missing_required_levels = missing_required_levels or []
        self.inconsistencies = inconsistencies or []


class HierarchyValidationError(ValidationError):
    """Exception raised when hierarchical data validation fails."""
    
    def __init__(self, message: str, hierarchy_level: Optional[str] = None,
                 parent_level: Optional[str] = None, child_level: Optional[str] = None,
                 invalid_relationships: Optional[List[Dict[str, Any]]] = None,
                 validation_type: str = 'consistency'):
        """
        Initialize hierarchy validation error.
        
        Args:
            message: Human-readable error message
            hierarchy_level: The hierarchical level where validation failed
            parent_level: Parent level in the hierarchy
            child_level: Child level in the hierarchy
            invalid_relationships: List of invalid parent-child relationships
            validation_type: Type of validation that failed (consistency, completeness, structure)
        """
        context = {
            'hierarchy_level': hierarchy_level,
            'parent_level': parent_level,
            'child_level': child_level,
            'invalid_relationships': invalid_relationships or [],
            'validation_type': validation_type
        }
        super().__init__(
            message=message,
            field_name=hierarchy_level,
            validation_rules=[f'hierarchy_{validation_type}']
        )
        self.context.update(context)
        self.hierarchy_level = hierarchy_level
        self.parent_level = parent_level
        self.child_level = child_level
        self.invalid_relationships = invalid_relationships or []
        self.validation_type = validation_type


class HierarchicalMappingError(MappingProcessError):
    """Exception raised when hierarchical code mapping fails."""
    
    def __init__(self, message: str, hierarchy_level: Optional[str] = None,
                 failed_mappings: Optional[int] = None, total_mappings: Optional[int] = None,
                 parent_context: Optional[Dict[str, Any]] = None,
                 fallback_available: bool = False,
                 original_error: Optional[Exception] = None):
        """
        Initialize hierarchical mapping error.
        
        Args:
            message: Human-readable error message
            hierarchy_level: The hierarchical level where mapping failed
            failed_mappings: Number of mappings that failed
            total_mappings: Total number of mappings attempted
            parent_context: Context information about parent level
            fallback_available: Whether a fallback strategy is available
            original_error: Original exception that caused this error
        """
        context = {
            'hierarchy_level': hierarchy_level,
            'failed_mappings': failed_mappings,
            'total_mappings': total_mappings,
            'success_rate': ((total_mappings - failed_mappings) / total_mappings * 100) 
                           if total_mappings and failed_mappings is not None else None,
            'parent_context': parent_context or {},
            'fallback_available': fallback_available,
            'original_error': str(original_error) if original_error else None,
            'original_error_type': type(original_error).__name__ if original_error else None
        }
        super().__init__(
            message=message,
            strategy=f'hierarchical_mapping_{hierarchy_level}' if hierarchy_level else 'hierarchical_mapping',
            entity_count=total_mappings,
            processed_count=(total_mappings - failed_mappings) if total_mappings and failed_mappings is not None else None,
            original_error=original_error
        )
        self.context.update(context)
        self.hierarchy_level = hierarchy_level
        self.failed_mappings = failed_mappings
        self.total_mappings = total_mappings
        self.parent_context = parent_context or {}
        self.fallback_available = fallback_available


# Utility functions for exception handling

def create_validation_error(field_name: str, value: Any, rules: List[str]) -> ValidationError:
    """
    Create a standardized validation error.
    
    Args:
        field_name: Name of the field that failed validation
        value: Invalid value
        rules: List of validation rules that were violated
        
    Returns:
        ValidationError instance
    """
    message = f"Validation failed for field '{field_name}' with value '{value}'"
    if rules:
        message += f". Violated rules: {', '.join(rules)}"
    
    return ValidationError(
        message=message,
        field_name=field_name,
        invalid_value=value,
        validation_rules=rules
    )


def create_file_error(operation: str, file_path: str, original_error: Exception) -> FileAccessError:
    """
    Create a standardized file access error.
    
    Args:
        operation: Type of file operation that failed
        file_path: Path to the file
        original_error: Original exception
        
    Returns:
        FileAccessError instance
    """
    message = f"Failed to {operation} file '{file_path}': {str(original_error)}"
    
    return FileAccessError(
        message=message,
        file_path=file_path,
        operation=operation,
        original_error=original_error
    )


def create_mapping_error(strategy: str, entity_data: Dict[str, Any], 
                        original_error: Exception) -> MatchingError:
    """
    Create a standardized matching error.
    
    Args:
        strategy: Name of the matching strategy
        entity_data: Data of the entity being processed
        original_error: Original exception
        
    Returns:
        MatchingError instance
    """
    entity_desc = f"{entity_data.get('district', 'N/A')}/{entity_data.get('block', 'N/A')}/{entity_data.get('village', 'N/A')}"
    message = f"Matching failed for entity {entity_desc} using {strategy} strategy: {str(original_error)}"
    
    return MatchingError(
        message=message,
        entity_data=entity_data,
        matcher_type=strategy,
        original_error=original_error
    )


def is_recoverable_error(error: Exception) -> bool:
    """
    Determine if an error is recoverable.
    
    Args:
        error: Exception to check
        
    Returns:
        True if the error is potentially recoverable, False otherwise
    """
    # File I/O errors are often recoverable with retry
    if isinstance(error, (FileAccessError, DataLoadError)):
        return True
    
    # Some mapping errors can be recovered by skipping problematic records
    if isinstance(error, (MatchingError, ValidationError)):
        return True
    
    # Configuration and critical system errors are usually not recoverable
    if isinstance(error, (ConfigurationError, RecoveryError)):
        return False
    
    # For unknown errors, assume they might be recoverable
    return True


def get_error_severity(error: Exception) -> str:
    """
    Get the severity level of an error.
    
    Args:
        error: Exception to evaluate
        
    Returns:
        Severity level string (low, medium, high, critical)
    """
    if isinstance(error, ConfigurationError):
        return 'critical'
    elif isinstance(error, (DataLoadError, FileAccessError)):
        return 'high'
    elif isinstance(error, HierarchyDetectionError):
        # Hierarchy detection failures are high severity as they block processing
        return 'high'
    elif isinstance(error, (MappingProcessError, OutputGenerationError, HierarchicalMappingError)):
        return 'medium'
    elif isinstance(error, (ValidationError, MatchingError, HierarchyValidationError)):
        return 'low'
    elif isinstance(error, DataQualityError):
        return error.severity
    else:
        return 'medium'


def create_hierarchy_detection_error(
    detected_levels: List[str],
    missing_required: List[str],
    inconsistencies: Optional[List[str]] = None
) -> HierarchyDetectionError:
    """
    Create a standardized hierarchy detection error.
    
    Args:
        detected_levels: List of hierarchical levels that were detected
        missing_required: List of required levels that are missing
        inconsistencies: Optional list of hierarchical inconsistencies
        
    Returns:
        HierarchyDetectionError instance
    """
    if missing_required:
        message = (
            f"Hierarchy detection failed: missing required levels {', '.join(missing_required)}. "
            f"Detected levels: {', '.join(detected_levels) if detected_levels else 'none'}"
        )
    elif inconsistencies:
        message = (
            f"Hierarchy detection found inconsistencies: {'; '.join(inconsistencies)}. "
            f"Detected levels: {', '.join(detected_levels)}"
        )
    else:
        message = f"Hierarchy detection failed with detected levels: {', '.join(detected_levels) if detected_levels else 'none'}"
    
    return HierarchyDetectionError(
        message=message,
        detected_levels=detected_levels,
        missing_required_levels=missing_required,
        inconsistencies=inconsistencies or []
    )


def create_hierarchy_validation_error(
    hierarchy_level: str,
    parent_level: Optional[str] = None,
    child_level: Optional[str] = None,
    invalid_count: int = 0,
    validation_type: str = 'consistency'
) -> HierarchyValidationError:
    """
    Create a standardized hierarchy validation error.
    
    Args:
        hierarchy_level: The hierarchical level where validation failed
        parent_level: Parent level in the hierarchy
        child_level: Child level in the hierarchy
        invalid_count: Number of invalid relationships found
        validation_type: Type of validation that failed
        
    Returns:
        HierarchyValidationError instance
    """
    if parent_level and child_level:
        message = (
            f"Hierarchical {validation_type} validation failed at level '{hierarchy_level}': "
            f"found {invalid_count} invalid relationships between {parent_level} and {child_level}"
        )
    else:
        message = (
            f"Hierarchical {validation_type} validation failed at level '{hierarchy_level}': "
            f"found {invalid_count} validation issues"
        )
    
    return HierarchyValidationError(
        message=message,
        hierarchy_level=hierarchy_level,
        parent_level=parent_level,
        child_level=child_level,
        validation_type=validation_type
    )


def create_hierarchical_mapping_error(
    hierarchy_level: str,
    failed_count: int,
    total_count: int,
    parent_context: Optional[Dict[str, Any]] = None,
    fallback_available: bool = False
) -> HierarchicalMappingError:
    """
    Create a standardized hierarchical mapping error.
    
    Args:
        hierarchy_level: The hierarchical level where mapping failed
        failed_count: Number of mappings that failed
        total_count: Total number of mappings attempted
        parent_context: Context information about parent level
        fallback_available: Whether a fallback strategy is available
        
    Returns:
        HierarchicalMappingError instance
    """
    success_rate = ((total_count - failed_count) / total_count * 100) if total_count > 0 else 0
    
    message = (
        f"Hierarchical code mapping failed at level '{hierarchy_level}': "
        f"{failed_count}/{total_count} mappings failed ({success_rate:.1f}% success rate)"
    )
    
    if parent_context:
        parent_info = ', '.join([f"{k}={v}" for k, v in parent_context.items()])
        message += f". Parent context: {parent_info}"
    
    if fallback_available:
        message += ". Fallback strategy available"
    
    return HierarchicalMappingError(
        message=message,
        hierarchy_level=hierarchy_level,
        failed_mappings=failed_count,
        total_mappings=total_count,
        parent_context=parent_context,
        fallback_available=fallback_available
    )