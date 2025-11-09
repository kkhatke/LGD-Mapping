"""
Configuration management for LGD mapping application.

This module provides dataclasses and utilities for managing application
configuration including file paths, mapping parameters, and processing options.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
from pathlib import Path


@dataclass
class MappingConfig:
    """Configuration class for LGD mapping parameters."""
    
    # Input file paths
    input_entities_file: str
    input_codes_file: str
    
    # Output configuration
    output_directory: str
    
    # Fuzzy matching thresholds (in descending order)
    fuzzy_thresholds: List[int] = field(default_factory=lambda: [95, 90])
    
    # District name to code mapping for standardization
    district_code_mapping: Dict[str, int] = field(default_factory=dict)
    
    # District code mapping configuration
    enable_district_code_mapping: bool = True
    district_fuzzy_threshold: int = 85
    
    # Hierarchical mapping configuration
    enable_hierarchical_mapping: bool = True
    enable_state_code_mapping: bool = False  # Usually not needed
    enable_block_code_mapping: bool = True
    enable_gp_code_mapping: bool = True
    
    # Level-specific fuzzy thresholds
    state_fuzzy_threshold: int = 85
    block_fuzzy_threshold: int = 90
    gp_fuzzy_threshold: int = 90
    village_fuzzy_threshold: int = 95
    
    # Hierarchy validation
    enforce_hierarchy_consistency: bool = True
    allow_partial_hierarchy: bool = True
    
    # Performance optimization settings
    chunk_size: Optional[int] = None
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_paths()
        self._validate_thresholds()
        self._ensure_output_directory()
    
    def _validate_paths(self):
        """Validate that input files exist."""
        if not os.path.exists(self.input_entities_file):
            raise FileNotFoundError(f"Entities file not found: {self.input_entities_file}")
        
        if not os.path.exists(self.input_codes_file):
            raise FileNotFoundError(f"Codes file not found: {self.input_codes_file}")
    
    def _validate_thresholds(self):
        """Validate fuzzy matching thresholds."""
        if not self.fuzzy_thresholds:
            raise ValueError("At least one fuzzy threshold must be specified")
        
        for threshold in self.fuzzy_thresholds:
            if not 0 <= threshold <= 100:
                raise ValueError(f"Fuzzy threshold must be between 0 and 100: {threshold}")
        
        # Ensure thresholds are in descending order for optimal processing
        if self.fuzzy_thresholds != sorted(self.fuzzy_thresholds, reverse=True):
            self.fuzzy_thresholds = sorted(self.fuzzy_thresholds, reverse=True)
        
        # Validate hierarchical fuzzy thresholds
        hierarchical_thresholds = [
            ('state', self.state_fuzzy_threshold),
            ('district', self.district_fuzzy_threshold),
            ('block', self.block_fuzzy_threshold),
            ('gp', self.gp_fuzzy_threshold),
            ('village', self.village_fuzzy_threshold)
        ]
        
        for level_name, threshold in hierarchical_thresholds:
            if not 0 <= threshold <= 100:
                raise ValueError(
                    f"{level_name.capitalize()} fuzzy threshold must be between 0 and 100: {threshold}"
                )
    
    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist."""
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MappingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'input_entities_file': self.input_entities_file,
            'input_codes_file': self.input_codes_file,
            'output_directory': self.output_directory,
            'fuzzy_thresholds': self.fuzzy_thresholds,
            'district_code_mapping': self.district_code_mapping,
            'enable_district_code_mapping': self.enable_district_code_mapping,
            'district_fuzzy_threshold': self.district_fuzzy_threshold,
            'enable_hierarchical_mapping': self.enable_hierarchical_mapping,
            'enable_state_code_mapping': self.enable_state_code_mapping,
            'enable_block_code_mapping': self.enable_block_code_mapping,
            'enable_gp_code_mapping': self.enable_gp_code_mapping,
            'state_fuzzy_threshold': self.state_fuzzy_threshold,
            'block_fuzzy_threshold': self.block_fuzzy_threshold,
            'gp_fuzzy_threshold': self.gp_fuzzy_threshold,
            'village_fuzzy_threshold': self.village_fuzzy_threshold,
            'enforce_hierarchy_consistency': self.enforce_hierarchy_consistency,
            'allow_partial_hierarchy': self.allow_partial_hierarchy,
            'chunk_size': self.chunk_size,
            'log_level': self.log_level,
            'log_file': self.log_file
        }


@dataclass
class ProcessingStats:
    """Statistics tracking for mapping process."""
    
    total_entities: int = 0
    exact_matches: int = 0
    fuzzy_matches: Dict[int, int] = field(default_factory=dict)
    unmatched: int = 0
    processing_time: float = 0.0
    
    def get_match_rate(self) -> float:
        """Calculate overall match rate percentage."""
        if self.total_entities == 0:
            return 0.0
        
        total_matched = self.exact_matches + sum(self.fuzzy_matches.values())
        return (total_matched / self.total_entities) * 100
    
    def get_fuzzy_match_rate(self, threshold: int) -> float:
        """Calculate match rate for specific fuzzy threshold."""
        if self.total_entities == 0:
            return 0.0
        
        fuzzy_count = self.fuzzy_matches.get(threshold, 0)
        return (fuzzy_count / self.total_entities) * 100