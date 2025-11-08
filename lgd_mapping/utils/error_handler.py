"""
Error handling utilities for LGD mapping application.

This module provides utilities for error handling, retry mechanisms,
and graceful degradation strategies.
"""

import time
import logging
import functools
from typing import Callable, Any, Optional, List, Dict, Type, Union
from pathlib import Path

from ..exceptions import (
    LGDMappingError, FileAccessError, DataLoadError, MappingProcessError,
    RecoveryError, is_recoverable_error, get_error_severity
)


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0,
                 retry_exceptions: Optional[List[Type[Exception]]] = None):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Factor to multiply delay by for exponential backoff
            retry_exceptions: List of exception types to retry on
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retry_exceptions = retry_exceptions or [
            FileAccessError, DataLoadError, ConnectionError, TimeoutError
        ]


class ErrorHandler:
    """Centralized error handling and recovery manager."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the error handler.
        
        Args:
            logger: Optional logger instance for error logging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.recovery_strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default recovery strategies."""
        self.recovery_strategies.update({
            'file_retry': self._file_retry_strategy,
            'skip_record': self._skip_record_strategy,
            'use_default': self._use_default_strategy,
            'partial_processing': self._partial_processing_strategy
        })
    
    def handle_error(self, error: Exception, context: Dict[str, Any], 
                    recovery_strategy: Optional[str] = None) -> Any:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: Exception that occurred
            context: Context information about the error
            recovery_strategy: Optional specific recovery strategy to use
            
        Returns:
            Recovery result or raises the error if not recoverable
        """
        # Log the error
        self._log_error(error, context)
        
        # Update error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Determine if error is recoverable
        if not is_recoverable_error(error):
            self.logger.error(f"Non-recoverable error: {error}")
            raise error
        
        # Apply recovery strategy
        if recovery_strategy and recovery_strategy in self.recovery_strategies:
            try:
                return self.recovery_strategies[recovery_strategy](error, context)
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy '{recovery_strategy}' failed: {recovery_error}")
                raise RecoveryError(
                    f"Recovery strategy '{recovery_strategy}' failed",
                    recovery_strategy=recovery_strategy,
                    original_error=error
                )
        
        # Default recovery based on error type
        return self._apply_default_recovery(error, context)
    
    def _log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error with appropriate level and context."""
        severity = get_error_severity(error)
        error_info = {
            'error_type': type(error).__name__,
            'message': str(error),
            'severity': severity,
            'context': context
        }
        
        if hasattr(error, 'to_dict'):
            error_info.update(error.to_dict())
        
        if severity == 'critical':
            self.logger.critical(f"Critical error: {error_info}")
        elif severity == 'high':
            self.logger.error(f"High severity error: {error_info}")
        elif severity == 'medium':
            self.logger.warning(f"Medium severity error: {error_info}")
        else:
            self.logger.info(f"Low severity error: {error_info}")
    
    def _apply_default_recovery(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Apply default recovery strategy based on error type."""
        if isinstance(error, (FileAccessError, DataLoadError)):
            return self._file_retry_strategy(error, context)
        elif isinstance(error, MappingProcessError):
            return self._partial_processing_strategy(error, context)
        else:
            # For unknown errors, try to continue with default values
            return self._use_default_strategy(error, context)
    
    def _file_retry_strategy(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Retry file operations with exponential backoff."""
        retry_config = context.get('retry_config', RetryConfig())
        operation = context.get('operation')
        
        if not operation:
            raise error
        
        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                self.logger.info(f"Retrying file operation (attempt {attempt}/{retry_config.max_attempts})")
                return operation()
            except Exception as retry_error:
                if attempt == retry_config.max_attempts:
                    raise RecoveryError(
                        f"File operation failed after {retry_config.max_attempts} attempts",
                        recovery_strategy='file_retry',
                        original_error=error,
                        recovery_attempts=attempt
                    )
                
                # Calculate delay with exponential backoff
                delay = min(
                    retry_config.base_delay * (retry_config.backoff_factor ** (attempt - 1)),
                    retry_config.max_delay
                )
                
                self.logger.info(f"Waiting {delay:.2f} seconds before retry")
                time.sleep(delay)
        
        raise error
    
    def _skip_record_strategy(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Skip problematic record and continue processing."""
        record_info = context.get('record_info', 'unknown record')
        self.logger.warning(f"Skipping {record_info} due to error: {error}")
        
        # Return a default/empty result to indicate skipped record
        return context.get('skip_result', None)
    
    def _use_default_strategy(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Use default values when encountering errors."""
        default_value = context.get('default_value')
        
        if default_value is not None:
            self.logger.info(f"Using default value due to error: {error}")
            return default_value
        
        # If no default provided, re-raise the error
        raise error
    
    def _partial_processing_strategy(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Continue with partial processing when possible."""
        processed_count = context.get('processed_count', 0)
        total_count = context.get('total_count', 0)
        
        if processed_count > 0:
            self.logger.warning(
                f"Partial processing completed: {processed_count}/{total_count} records processed"
            )
            return context.get('partial_results', [])
        
        raise error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        total_errors = sum(self.error_counts.values())
        
        return {
            'total_errors': total_errors,
            'error_counts': self.error_counts.copy(),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] 
                                if self.error_counts else None
        }
    
    def reset_error_counts(self):
        """Reset error counters."""
        self.error_counts.clear()
        self.logger.info("Error counters reset")


def with_retry(retry_config: Optional[RetryConfig] = None, 
               logger: Optional[logging.Logger] = None):
    """
    Decorator for adding retry functionality to functions.
    
    Args:
        retry_config: Configuration for retry behavior
        logger: Optional logger for retry messages
        
    Returns:
        Decorated function with retry capability
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if this exception type should be retried
                    if not any(isinstance(e, exc_type) for exc_type in retry_config.retry_exceptions):
                        logger.debug(f"Exception {type(e).__name__} not in retry list, not retrying")
                        raise e
                    
                    if attempt == retry_config.max_attempts:
                        logger.error(f"Function {func.__name__} failed after {retry_config.max_attempts} attempts")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        retry_config.base_delay * (retry_config.backoff_factor ** (attempt - 1)),
                        retry_config.max_delay
                    )
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}/{retry_config.max_attempts}): {e}. "
                        f"Retrying in {delay:.2f} seconds"
                    )
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def with_error_handling(error_handler: Optional[ErrorHandler] = None,
                       recovery_strategy: Optional[str] = None,
                       context_provider: Optional[Callable] = None):
    """
    Decorator for adding error handling to functions.
    
    Args:
        error_handler: ErrorHandler instance to use
        recovery_strategy: Default recovery strategy to apply
        context_provider: Function to provide context for error handling
        
    Returns:
        Decorated function with error handling
    """
    if error_handler is None:
        error_handler = ErrorHandler()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get context information
                context = {}
                if context_provider:
                    try:
                        context = context_provider(*args, **kwargs)
                    except Exception as ctx_error:
                        error_handler.logger.warning(f"Error getting context: {ctx_error}")
                
                # Add function information to context
                context.update({
                    'function_name': func.__name__,
                    'args': str(args)[:200],  # Limit length for logging
                    'kwargs': str(kwargs)[:200]
                })
                
                # Handle the error
                return error_handler.handle_error(e, context, recovery_strategy)
        
        return wrapper
    return decorator


def safe_file_operation(operation: Callable, file_path: Union[str, Path], 
                       operation_name: str, retry_config: Optional[RetryConfig] = None,
                       logger: Optional[logging.Logger] = None) -> Any:
    """
    Safely perform file operations with retry and error handling.
    
    Args:
        operation: Function to perform the file operation
        file_path: Path to the file
        operation_name: Name of the operation for logging
        retry_config: Configuration for retry behavior
        logger: Optional logger instance
        
    Returns:
        Result of the file operation
        
    Raises:
        FileAccessError: If the operation fails after all retries
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if retry_config is None:
        retry_config = RetryConfig()
    
    file_path = Path(file_path)
    last_exception = None
    
    for attempt in range(1, retry_config.max_attempts + 1):
        try:
            logger.debug(f"Attempting {operation_name} on {file_path} (attempt {attempt})")
            return operation()
        
        except (OSError, IOError, PermissionError) as e:
            last_exception = e
            
            if attempt == retry_config.max_attempts:
                logger.error(f"{operation_name} failed after {retry_config.max_attempts} attempts: {e}")
                raise FileAccessError(
                    f"Failed to {operation_name} file after {retry_config.max_attempts} attempts",
                    file_path=str(file_path),
                    operation=operation_name,
                    original_error=e
                )
            
            # Calculate delay
            delay = min(
                retry_config.base_delay * (retry_config.backoff_factor ** (attempt - 1)),
                retry_config.max_delay
            )
            
            logger.warning(f"{operation_name} failed (attempt {attempt}): {e}. Retrying in {delay:.2f} seconds")
            time.sleep(delay)
        
        except Exception as e:
            # For non-file-related exceptions, don't retry
            logger.error(f"Non-recoverable error during {operation_name}: {e}")
            raise FileAccessError(
                f"Non-recoverable error during {operation_name}",
                file_path=str(file_path),
                operation=operation_name,
                original_error=e
            )
    
    # This should never be reached
    raise FileAccessError(
        f"Unexpected error during {operation_name}",
        file_path=str(file_path),
        operation=operation_name,
        original_error=last_exception
    )


def create_error_context(operation: str, **kwargs) -> Dict[str, Any]:
    """
    Create standardized error context dictionary.
    
    Args:
        operation: Name of the operation being performed
        **kwargs: Additional context information
        
    Returns:
        Dictionary with error context information
    """
    context = {
        'operation': operation,
        'timestamp': time.time(),
    }
    context.update(kwargs)
    return context


def log_error_details(logger: logging.Logger, error: Exception, 
                     context: Optional[Dict[str, Any]] = None):
    """
    Log detailed error information.
    
    Args:
        logger: Logger instance to use
        error: Exception to log
        context: Optional context information
    """
    error_details = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'severity': get_error_severity(error)
    }
    
    if hasattr(error, 'to_dict'):
        error_details.update(error.to_dict())
    
    if context:
        error_details['context'] = context
    
    # Log with appropriate level based on severity
    severity = error_details.get('severity', 'medium')
    if severity == 'critical':
        logger.critical(f"Critical error occurred: {error_details}")
    elif severity == 'high':
        logger.error(f"High severity error: {error_details}")
    elif severity == 'medium':
        logger.warning(f"Medium severity error: {error_details}")
    else:
        logger.info(f"Low severity error: {error_details}")