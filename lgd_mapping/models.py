"""
Data models for LGD mapping application.

This module defines the core data structures used throughout the mapping process.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from .utils.data_utils import safe_string_conversion, safe_int_conversion, is_null_or_empty

if TYPE_CHECKING:
    from .hierarchy.hierarchy_config import HierarchyConfiguration


@dataclass
class EntityRecord:
    """Represents a source entity record with flexible hierarchical administrative data."""
    
    # Core fields (always present)
    district: str
    block: str
    village: str
    
    # Optional hierarchical fields
    state: Optional[str] = None
    gp: Optional[str] = None
    subdistrict: Optional[str] = None
    
    # Code fields
    state_code: Optional[int] = None
    district_code: Optional[int] = None
    subdistrict_code: Optional[int] = None
    block_code: Optional[int] = None
    gp_code: Optional[int] = None
    village_code: Optional[int] = None
    
    # Metadata
    hierarchy_depth: int = 3  # Number of levels present
    hierarchy_levels: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Clean and validate data after initialization."""
        # Clean string fields using utility functions
        self.district = safe_string_conversion(self.district)
        self.block = safe_string_conversion(self.block)
        self.village = safe_string_conversion(self.village)
        
        # Clean optional hierarchical fields
        if self.state is not None:
            self.state = safe_string_conversion(self.state)
        if self.gp is not None:
            self.gp = safe_string_conversion(self.gp)
        if self.subdistrict is not None:
            self.subdistrict = safe_string_conversion(self.subdistrict)
        
        # Convert code fields if provided
        if self.state_code is not None:
            self.state_code = safe_int_conversion(self.state_code)
        if self.district_code is not None:
            self.district_code = safe_int_conversion(self.district_code)
        if self.subdistrict_code is not None:
            self.subdistrict_code = safe_int_conversion(self.subdistrict_code)
        if self.block_code is not None:
            self.block_code = safe_int_conversion(self.block_code)
        if self.gp_code is not None:
            self.gp_code = safe_int_conversion(self.gp_code)
        if self.village_code is not None:
            self.village_code = safe_int_conversion(self.village_code)
        
        # Auto-calculate hierarchy depth and levels if not set
        if not self.hierarchy_levels:
            self.hierarchy_levels = self._detect_hierarchy_levels()
            self.hierarchy_depth = len(self.hierarchy_levels)
    
    def _detect_hierarchy_levels(self) -> List[str]:
        """Detect which hierarchical levels are present in this record."""
        levels = []
        
        # Check each level in hierarchical order
        if not is_null_or_empty(self.state):
            levels.append('state')
        if not is_null_or_empty(self.district):
            levels.append('district')
        if not is_null_or_empty(self.subdistrict):
            levels.append('subdistrict')
        if not is_null_or_empty(self.block):
            levels.append('block')
        if not is_null_or_empty(self.gp):
            levels.append('gp')
        if not is_null_or_empty(self.village):
            levels.append('village')
        
        return levels
    
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
    
    def get_hierarchical_path(self) -> List[Tuple[str, str]]:
        """
        Get list of (level_name, value) tuples in hierarchical order.
        
        Returns:
            List of tuples containing level name and corresponding value
            
        Example:
            [('state', 'Karnataka'), ('district', 'Bangalore'), ('block', 'Anekal'), 
             ('village', 'Jigani')]
        """
        path = []
        
        # Add levels in hierarchical order
        if not is_null_or_empty(self.state):
            path.append(('state', self.state))
        if not is_null_or_empty(self.district):
            path.append(('district', self.district))
        if not is_null_or_empty(self.subdistrict):
            path.append(('subdistrict', self.subdistrict))
        if not is_null_or_empty(self.block):
            path.append(('block', self.block))
        if not is_null_or_empty(self.gp):
            path.append(('gp', self.gp))
        if not is_null_or_empty(self.village):
            path.append(('village', self.village))
        
        return path
    
    def get_hierarchical_codes(self) -> List[Tuple[str, int]]:
        """
        Get list of (level_name, code) tuples in hierarchical order.
        
        Returns:
            List of tuples containing level name and corresponding code.
            Only includes levels where codes are available.
            
        Example:
            [('state', 29), ('district', 560), ('block', 5601), ('village', 123456)]
        """
        codes = []
        
        # Add codes in hierarchical order (only if not None)
        if self.state_code is not None:
            codes.append(('state_code', self.state_code))
        if self.district_code is not None:
            codes.append(('district_code', self.district_code))
        if self.block_code is not None:
            codes.append(('block_code', self.block_code))
        if self.gp_code is not None:
            codes.append(('gp_code', self.gp_code))
        if self.village_code is not None:
            codes.append(('village_code', self.village_code))
        
        return codes


@dataclass
class LGDRecord:
    """Represents an LGD code record with flexible hierarchical administrative data."""
    
    # Core fields (always present)
    district_code: int
    district: str
    block_code: int
    block: str
    village_code: int
    village: str
    
    # Optional hierarchical fields
    state_code: Optional[int] = None
    state: Optional[str] = None
    gp_code: Optional[int] = None
    gp: Optional[str] = None
    subdistrict_code: Optional[int] = None
    subdistrict: Optional[str] = None
    
    def __post_init__(self):
        """Clean and validate data after initialization."""
        # Clean string fields using utility functions
        self.district = safe_string_conversion(self.district)
        self.block = safe_string_conversion(self.block)
        self.village = safe_string_conversion(self.village)
        
        # Clean optional hierarchical fields
        if self.state is not None:
            self.state = safe_string_conversion(self.state)
        if self.gp is not None:
            self.gp = safe_string_conversion(self.gp)
        if self.subdistrict is not None:
            self.subdistrict = safe_string_conversion(self.subdistrict)
        
        # Validate and convert numeric fields
        self.district_code = safe_int_conversion(self.district_code)
        self.block_code = safe_int_conversion(self.block_code)
        self.village_code = safe_int_conversion(self.village_code)
        
        # Convert optional code fields
        if self.state_code is not None:
            self.state_code = safe_int_conversion(self.state_code)
        if self.gp_code is not None:
            self.gp_code = safe_int_conversion(self.gp_code)
        if self.subdistrict_code is not None:
            self.subdistrict_code = safe_int_conversion(self.subdistrict_code)
    
    def create_uid(self) -> str:
        """
        Create unique identifier for this LGD record (backward compatible).
        
        Returns:
            3-level UID in format: district_code_block_code_village_code
        """
        return f"{self.district_code}_{self.block_code}_{self.village_code}"
    
    def create_hierarchical_uid(
        self, 
        hierarchy_config: Optional['HierarchyConfiguration'] = None
    ) -> str:
        """
        Create hierarchical UID based on available levels and configuration.
        
        Args:
            hierarchy_config: Optional hierarchy configuration to determine which
                            levels to include. If None, includes all available levels.
        
        Returns:
            Hierarchical UID with components separated by underscores
            
        Example:
            With full hierarchy: "29_560_5601_560101_123456"
            With partial hierarchy: "560_5601_123456"
        """
        components = []
        
        # Define the standard hierarchy order
        hierarchy_order = [
            ('state_code', self.state_code),
            ('district_code', self.district_code),
            ('subdistrict_code', self.subdistrict_code),
            ('block_code', self.block_code),
            ('gp_code', self.gp_code),
            ('village_code', self.village_code)
        ]
        
        # If hierarchy_config is provided, use detected levels
        if hierarchy_config is not None:
            detected_level_names = set(hierarchy_config.detected_levels)
            for level_name, code_value in hierarchy_order:
                # Extract base level name (remove '_code' suffix)
                base_level = level_name.replace('_code', '')
                if base_level in detected_level_names and code_value is not None:
                    components.append(str(code_value))
        else:
            # Include all available codes
            for level_name, code_value in hierarchy_order:
                if code_value is not None:
                    components.append(str(code_value))
        
        return '_'.join(components) if components else self.create_uid()
    
    def create_block_uid(self) -> str:
        """Create block-level unique identifier."""
        return f"{self.district_code}_{self.block}"
    
    def validate_hierarchy(self) -> Tuple[bool, List[str]]:
        """
        Validate hierarchical consistency of this record.
        
        Checks that child levels have corresponding parent levels and that
        the hierarchical structure is logically consistent.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
            
        Example:
            (True, []) - Valid hierarchy
            (False, ['GP code present but block code missing']) - Invalid
        """
        issues = []
        
        # Check that if a child level is present, parent levels are also present
        
        # If GP is present, block must be present
        if (self.gp_code is not None or not is_null_or_empty(self.gp)):
            if self.block_code is None or is_null_or_empty(self.block):
                issues.append("GP level present but block level is missing")
        
        # If subdistrict is present, district must be present
        if (self.subdistrict_code is not None or not is_null_or_empty(self.subdistrict)):
            if self.district_code is None or is_null_or_empty(self.district):
                issues.append("Subdistrict level present but district level is missing")
        
        # If block is present, district must be present (always true for valid records)
        if self.block_code is not None:
            if self.district_code is None:
                issues.append("Block code present but district code is missing")
        
        # If district is present, state should ideally be present (warning, not error)
        # This is informational only
        
        # Check that codes and names are consistent (both present or both absent)
        if (self.state_code is not None) != (not is_null_or_empty(self.state)):
            issues.append("State code and state name inconsistency")
        
        if (self.gp_code is not None) != (not is_null_or_empty(self.gp)):
            issues.append("GP code and GP name inconsistency")
        
        if (self.subdistrict_code is not None) != (not is_null_or_empty(self.subdistrict)):
            issues.append("Subdistrict code and subdistrict name inconsistency")
        
        return len(issues) == 0, issues
    
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
    """Represents the result of mapping an entity to LGD codes with hierarchical information."""
    
    entity: EntityRecord
    lgd_match: Optional[LGDRecord]
    match_type: str  # 'exact', 'fuzzy_95', 'fuzzy_90', 'unmatched'
    match_score: Optional[float] = None
    alternative_matches: List[str] = field(default_factory=list)
    
    # Hierarchical matching information
    hierarchy_match_details: Dict[str, Any] = field(default_factory=dict)
    # Example: {'state': 'exact', 'district': 'exact', 'block': 'fuzzy_90', 'village': 'exact'}
    
    hierarchy_confidence: float = 0.0  # 0-1 score based on hierarchy alignment
    
    def __post_init__(self):
        """Validate match type and calculate hierarchy confidence after initialization."""
        valid_types = ['exact', 'unmatched']
        # Allow fuzzy match types with any threshold (fuzzy_XX format)
        if (self.match_type not in valid_types and 
            not (self.match_type.startswith('fuzzy_') and 
                 self.match_type[6:].isdigit())):
            raise ValueError(f"Invalid match_type: {self.match_type}. "
                           f"Must be 'exact', 'unmatched', or 'fuzzy_XX' where XX is a number")
        
        # Auto-calculate hierarchy confidence if not set and match details exist
        if self.hierarchy_confidence == 0.0 and self.hierarchy_match_details:
            self.hierarchy_confidence = self.calculate_hierarchy_confidence()
    
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
    
    def get_hierarchy_match_summary(self) -> str:
        """
        Get human-readable summary of hierarchical matching.
        
        Returns:
            String describing how each hierarchical level was matched
            
        Example:
            "state: exact, district: exact, block: fuzzy (90%), village: exact"
            "district: exact, block: fuzzy (85%), village: exact (3-level hierarchy)"
        """
        if not self.hierarchy_match_details:
            return "No hierarchical match details available"
        
        summary_parts = []
        
        # Process each level in hierarchical order
        hierarchy_order = ['state', 'district', 'subdistrict', 'block', 'gp', 'village']
        
        for level in hierarchy_order:
            if level in self.hierarchy_match_details:
                match_info = self.hierarchy_match_details[level]
                
                if match_info == 'exact':
                    summary_parts.append(f"{level}: exact")
                elif match_info == 'unmatched':
                    summary_parts.append(f"{level}: unmatched")
                elif isinstance(match_info, str) and match_info.startswith('fuzzy_'):
                    # Extract threshold from fuzzy_XX format
                    try:
                        threshold = match_info.split('_')[1]
                        summary_parts.append(f"{level}: fuzzy ({threshold}%)")
                    except (IndexError, ValueError):
                        summary_parts.append(f"{level}: {match_info}")
                else:
                    summary_parts.append(f"{level}: {match_info}")
        
        # Add hierarchy depth information
        depth = len(self.hierarchy_match_details)
        summary = ", ".join(summary_parts)
        
        if depth <= 3:
            summary += f" ({depth}-level hierarchy)"
        
        return summary
    
    def calculate_hierarchy_confidence(self) -> float:
        """
        Calculate confidence score based on hierarchical alignment.
        
        The confidence score is calculated by:
        1. Assigning weights to each hierarchical level (higher levels = more weight)
        2. Scoring each level based on match type (exact=1.0, fuzzy=threshold/100, unmatched=0.0)
        3. Computing weighted average
        
        Returns:
            Confidence score between 0.0 and 1.0
            
        Example:
            All exact matches: 1.0
            Mix of exact and fuzzy_90: ~0.95
            Some unmatched levels: <0.8
        """
        if not self.hierarchy_match_details:
            # Fall back to basic match type
            if self.match_type == 'exact':
                return 1.0
            elif self.match_type.startswith('fuzzy_'):
                try:
                    threshold = int(self.match_type[6:])
                    return threshold / 100.0
                except (ValueError, IndexError):
                    return 0.5
            else:
                return 0.0
        
        # Define weights for each hierarchical level (higher = more important)
        level_weights = {
            'state': 0.10,
            'district': 0.25,
            'subdistrict': 0.15,
            'block': 0.20,
            'gp': 0.10,
            'village': 0.30
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for level, match_info in self.hierarchy_match_details.items():
            weight = level_weights.get(level, 0.15)  # Default weight for unknown levels
            total_weight += weight
            
            # Calculate score for this level
            if match_info == 'exact':
                level_score = 1.0
            elif match_info == 'unmatched':
                level_score = 0.0
            elif isinstance(match_info, str) and match_info.startswith('fuzzy_'):
                # Extract threshold from fuzzy_XX format
                try:
                    threshold = int(match_info.split('_')[1])
                    level_score = threshold / 100.0
                except (IndexError, ValueError):
                    level_score = 0.5
            else:
                # Unknown match type, use conservative score
                level_score = 0.5
            
            weighted_score += weight * level_score
        
        # Calculate weighted average
        if total_weight > 0:
            confidence = weighted_score / total_weight
        else:
            confidence = 0.0
        
        return round(confidence, 3)