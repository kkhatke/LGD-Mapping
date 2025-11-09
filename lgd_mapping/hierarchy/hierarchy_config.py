"""
Hierarchy configuration for LGD mapping application.

This module defines the data structures for representing and managing
hierarchical administrative structures with flexible levels.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HierarchyLevel:
    """
    Represents a single level in the administrative hierarchy.
    
    Attributes:
        name: Level identifier (e.g., 'state', 'district', 'block', 'gp', 'village')
        code_column: Name of the column containing codes for this level
        name_column: Name of the column containing names for this level
        is_required: Whether this level must be present in the data
        parent_level: Name of the parent level in the hierarchy (None for top level)
        fuzzy_threshold: Fuzzy matching threshold specific to this level (0-100)
    """
    name: str
    code_column: str
    name_column: str
    is_required: bool
    parent_level: Optional[str]
    fuzzy_threshold: Optional[int] = None
    
    def __post_init__(self):
        """Validate hierarchy level configuration."""
        if self.fuzzy_threshold is not None:
            if not 0 <= self.fuzzy_threshold <= 100:
                raise ValueError(
                    f"Fuzzy threshold for {self.name} must be between 0 and 100: "
                    f"{self.fuzzy_threshold}"
                )


@dataclass
class HierarchyConfiguration:
    """
    Configuration for hierarchical mapping across administrative levels.
    
    Attributes:
        levels: List of all possible hierarchy levels in order
        detected_levels: List of level names actually present in the data
        enable_code_mapping: Dictionary controlling code mapping per level
        fuzzy_thresholds: Dictionary of fuzzy matching thresholds per level
    """
    levels: List[HierarchyLevel]
    detected_levels: List[str] = field(default_factory=list)
    enable_code_mapping: Dict[str, bool] = field(default_factory=dict)
    fuzzy_thresholds: Dict[str, int] = field(default_factory=dict)
    
    def get_level(self, name: str) -> Optional[HierarchyLevel]:
        """
        Get hierarchy level by name.
        
        Args:
            name: Name of the level to retrieve
            
        Returns:
            HierarchyLevel object if found, None otherwise
        """
        for level in self.levels:
            if level.name == name:
                return level
        return None
    
    def get_parent_level(self, level_name: str) -> Optional[HierarchyLevel]:
        """
        Get the parent level of a given level.
        
        Args:
            level_name: Name of the level whose parent to find
            
        Returns:
            Parent HierarchyLevel object if found, None otherwise
        """
        level = self.get_level(level_name)
        if level is None or level.parent_level is None:
            return None
        return self.get_level(level.parent_level)
    
    def get_child_level(self, level_name: str) -> Optional[HierarchyLevel]:
        """
        Get the child level of a given level.
        
        Args:
            level_name: Name of the level whose child to find
            
        Returns:
            Child HierarchyLevel object if found, None otherwise
        """
        for level in self.levels:
            if level.parent_level == level_name:
                return level
        return None
    
    def is_level_present(self, level_name: str) -> bool:
        """
        Check if a level is present in the detected hierarchy.
        
        Args:
            level_name: Name of the level to check
            
        Returns:
            True if level is present, False otherwise
        """
        return level_name in self.detected_levels
    
    def get_hierarchy_depth(self) -> int:
        """
        Get the depth of the detected hierarchy.
        
        Returns:
            Number of levels in the detected hierarchy
        """
        return len(self.detected_levels)
    
    def get_detected_levels_in_order(self) -> List[HierarchyLevel]:
        """
        Get detected levels as HierarchyLevel objects in hierarchical order.
        
        Returns:
            List of HierarchyLevel objects for detected levels
        """
        detected = []
        for level in self.levels:
            if level.name in self.detected_levels:
                detected.append(level)
        return detected
    
    def get_fuzzy_threshold(self, level_name: str) -> Optional[int]:
        """
        Get fuzzy matching threshold for a specific level.
        
        Args:
            level_name: Name of the level
            
        Returns:
            Fuzzy threshold value or None if not configured
        """
        # First check custom thresholds
        if level_name in self.fuzzy_thresholds:
            return self.fuzzy_thresholds[level_name]
        
        # Fall back to level default
        level = self.get_level(level_name)
        if level:
            return level.fuzzy_threshold
        
        return None
    
    def is_code_mapping_enabled(self, level_name: str) -> bool:
        """
        Check if code mapping is enabled for a specific level.
        
        Args:
            level_name: Name of the level
            
        Returns:
            True if code mapping is enabled, False otherwise
        """
        return self.enable_code_mapping.get(level_name, True)
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate the hierarchy configuration.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check that detected levels exist in defined levels
        defined_level_names = {level.name for level in self.levels}
        for detected in self.detected_levels:
            if detected not in defined_level_names:
                issues.append(f"Detected level '{detected}' not found in defined levels")
        
        # Check that required levels are present
        for level in self.levels:
            if level.is_required and level.name not in self.detected_levels:
                issues.append(f"Required level '{level.name}' not detected in data")
        
        # Check parent-child relationships
        for level in self.levels:
            if level.parent_level is not None:
                parent = self.get_level(level.parent_level)
                if parent is None:
                    issues.append(
                        f"Level '{level.name}' references non-existent parent '{level.parent_level}'"
                    )
        
        # Check for circular dependencies
        visited = set()
        for level in self.levels:
            current = level
            path = []
            while current is not None:
                if current.name in visited:
                    break
                if current.name in path:
                    issues.append(f"Circular dependency detected in hierarchy: {' -> '.join(path)}")
                    break
                path.append(current.name)
                current = self.get_parent_level(current.name)
            visited.update(path)
        
        return len(issues) == 0, issues
