"""
Logging configuration for LGD mapping application.

This module provides comprehensive logging infrastructure with configurable
levels, file output, and progress tracking capabilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class MappingLogger:
    """Custom logger for LGD mapping operations."""
    
    def __init__(self, name: str = "lgd_mapping", level: str = "INFO", 
                 log_file: Optional[str] = None):
        """
        Initialize the mapping logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for log output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            self._setup_file_handler(log_file, formatter)
    
    def _setup_file_handler(self, log_file: str, formatter: logging.Formatter):
        """Set up file logging handler."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_processing_start(self, entities_count: int, codes_count: int):
        """Log the start of processing with data counts."""
        self.info("=" * 60)
        self.info("LGD MAPPING PROCESS STARTED")
        self.info("=" * 60)
        self.info(f"Processing {entities_count:,} entity records")
        self.info(f"Using {codes_count:,} LGD code records")
        self.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log_processing_complete(self, stats):
        """Log processing completion with statistics."""
        self.info("=" * 60)
        self.info("LGD MAPPING PROCESS COMPLETED")
        self.info("=" * 60)
        self.info(f"Total entities processed: {stats.total_entities:,}")
        self.info(f"Exact matches: {stats.exact_matches:,}")
        
        for threshold, count in stats.fuzzy_matches.items():
            self.info(f"Fuzzy matches ({threshold}%): {count:,}")
        
        self.info(f"Unmatched records: {stats.unmatched:,}")
        self.info(f"Overall match rate: {stats.get_match_rate():.2f}%")
        self.info(f"Processing time: {stats.processing_time:.2f} seconds")
        self.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log_phase_start(self, phase_name: str):
        """Log the start of a processing phase."""
        self.info("-" * 40)
        self.info(f"Starting {phase_name}")
        self.info("-" * 40)
    
    def log_phase_complete(self, phase_name: str, count: int, duration: float):
        """Log the completion of a processing phase."""
        self.info(f"Completed {phase_name}")
        self.info(f"Records processed: {count:,}")
        self.info(f"Duration: {duration:.2f} seconds")
    
    def log_match_statistics(self, strategy: str, matched: int, total: int, 
                           duration: float):
        """Log matching statistics for a specific strategy."""
        match_rate = (matched / total * 100) if total > 0 else 0
        self.info(f"{strategy} - Matched: {matched:,}/{total:,} ({match_rate:.2f}%) "
                 f"in {duration:.2f}s")
    
    def log_data_quality_warning(self, message: str):
        """Log data quality warnings."""
        self.warning(f"DATA QUALITY: {message}")
    
    def log_file_operation(self, operation: str, file_path: str, record_count: int):
        """Log file operations."""
        self.info(f"{operation}: {file_path} ({record_count:,} records)")


def setup_logging(config) -> MappingLogger:
    """
    Set up logging based on configuration.
    
    Args:
        config: MappingConfig instance
        
    Returns:
        Configured MappingLogger instance
    """
    log_file = None
    if config.log_file:
        log_file = config.log_file
    elif config.output_directory:
        # Create default log file in output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(config.output_directory) / f"mapping_log_{timestamp}.txt"
    
    return MappingLogger(
        name="lgd_mapping",
        level=config.log_level,
        log_file=str(log_file) if log_file else None
    )