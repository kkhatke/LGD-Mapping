"""
Data models for LGD mapping application.

This module defines the core data structures used throughout the mapping process.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from .utils.data_utils import safe_string_conversion, safe_int_conversion, is_null_or_empty


@dataclass
class EntityRecord:
    """Represents a source entity record with district, block, and village."""
    
    district: str
    block: str
    village: str
    district_code: Optional[int] = None
    
    def __post_init__(self):
        """Clean and validate data after initialization."""
        # Clean string fields using utility functions
        self.district = safe_string_conversion(self.district)
        self.block = safe_string_conversion(self.block)
        self.village = safe_string_conversion(self.village)
        
        # Convert district_code if provided
        if self.district_code is not None:
            self.district_code = safe_int_conversion(self.district_code)
    
    def is_valid(self) -> bool:
        """Check if the record has all required fields."""
        return not any([
            is_null_or_empty(self.district),
            is_null_or_empty(self.block),
            is_null_or_empty(self.village)
        ])
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors for this record."""
        errors = []
        
        if is_null_or_empty(self.district):
            errors.append("District is required")
        if is_null_or_empty(self.block):
            errors.append("Block is required")
        if is_null_or_empty(self.village):
            errors.append("Village is required")
            
        return errors


@dataclass
class LGDRecord:
    """Represents an LGD code record with hierarchical administrative data."""
    
    district_code: int
    district: str
    block_code: int
    block: str
    village_code: int
    village: str
    gp_code: Optional[int] = None
    gp: Optional[str] = None
    
    def __post_init__(self):
        """Clean and validate data after initialization."""
        # Clean string fields using utility functions
        self.district = safe_string_conversion(self.district)
        self.block = safe_string_conversion(self.block)
        self.village = safe_string_conversion(self.village)
        self.gp = safe_string_conversion(self.gp) if self.gp else None
        
        # Validate and convert numeric fields
        self.district_code = safe_int_conversion(self.district_code)
        self.block_code = safe_int_conversion(self.block_code)
        self.village_code = safe_int_conversion(self.village_code)
        if self.gp_code is not None:
            self.gp_code = safe_int_conversion(self.gp_code)
    
    def create_uid(self) -> str:
        """Create unique identifier for this LGD record."""
        return f"{self.district_code}_{self.block_code}_{self.village_code}"
    
    def create_block_uid(self) -> str:
        """Create block-level unique identifier."""
        return f"{self.district_code}_{self.block}"
    
    def is_valid(self) -> bool:
        """Check if the record has all required fields and valid data types."""
        return not any([
            self.district_code is None,
            self.block_code is None,
            self.village_code is None,
            is_null_or_empty(self.district),
            is_null_or_empty(self.block),
            is_null_or_empty(self.village)
        ])
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors for this record."""
        errors = []
        
        if self.district_code is None:
            errors.append("District code is required")
        if self.block_code is None:
            errors.append("Block code is required")
        if self.village_code is None:
            errors.append("Village code is required")
        if is_null_or_empty(self.district):
            errors.append("District name is required")
        if is_null_or_empty(self.block):
            errors.append("Block name is required")
        if is_null_or_empty(self.village):
            errors.append("Village name is required")
            
        return errors


@dataclass
class MappingResult:
    """Represents the result of mapping an entity to LGD codes."""
    
    entity: EntityRecord
    lgd_match: Optional[LGDRecord]
    match_type: str  # 'exact', 'fuzzy_95', 'fuzzy_90', 'unmatched'
    match_score: Optional[float] = None
    alternative_matches: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate match type after initialization."""
        valid_types = ['exact', 'unmatched']
        # Allow fuzzy match types with any threshold (fuzzy_XX format)
        if (self.match_type not in valid_types and 
            not (self.match_type.startswith('fuzzy_') and 
                 self.match_type[6:].isdigit())):
            raise ValueError(f"Invalid match_type: {self.match_type}. "
                           f"Must be 'exact', 'unmatched', or 'fuzzy_XX' where XX is a number")
    
    def is_matched(self) -> bool:
        """Check if the entity was successfully matched."""
        return self.lgd_match is not None and self.match_type != 'unmatched'
    
    def get_confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.match_type == 'exact':
            return 'High'
        elif self.match_type.startswith('fuzzy_'):
            # Extract threshold from fuzzy_XX format
            try:
                threshold = int(self.match_type[6:])
                if threshold >= 95:
                    return 'High'
                elif threshold >= 85:
                    return 'Medium'
                else:
                    return 'Low'
            except (ValueError, IndexError):
                return 'Unknown'
        else:
            return 'None'