"""
Hierarchical code mapper for LGD mapping application.

This module provides functionality to map administrative names to codes across
all hierarchical levels (state, district, block, GP, village) using both exact
and fuzzy matching strategies with parent-level scoping.
"""

import logging
import re
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from rapidfuzz import fuzz

from lgd_mapping.hierarchy.hierarchy_config import HierarchyConfiguration, HierarchyLevel
from lgd_mapping.exceptions import (
    ValidationError,
    HierarchicalMappingError,
    create_hierarchical_mapping_error
)


class HierarchicalCodeMapper:
    """
    Maps administrative names to codes across all hierarchy levels.
    
    This class orchestrates code mapping for all detected hierarchical levels,
    using parent-level scoping to improve accuracy and reduce false matches.
    Supports both exact matching (after normalization) and fuzzy matching.
    """
    
    def __init__(
        self, 
        hierarchy_config: HierarchyConfiguration,
        logger: Optional[logging.Logger] = None,
        enable_caching: bool = True,
        cache_size_limit_mb: int = 100
    ):
        """
        Initialize the hierarchical code mapper.
        
        Args:
            hierarchy_config: Configuration describing the detected hierarchy
            logger: Optional logger instance for logging mapping operations
            enable_caching: Enable hierarchical lookup caching for performance
            cache_size_limit_mb: Maximum cache size in megabytes (default: 100MB)
        """
        self.hierarchy_config = hierarchy_config
        self.logger = logger or logging.getLogger(__name__)
        self.level_mappers = {}  # Cache of level-specific lookup dictionaries
        
        # Performance optimization settings
        self.enable_caching = enable_caching
        self.cache_size_limit_mb = cache_size_limit_mb
        self._lookup_cache = {}  # Cache for lookup results
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_size_bytes = 0
    
    def map_all_levels(
        self,
        entities_df: pd.DataFrame,
        lgd_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Map codes for all hierarchical levels in order from top to bottom.
        
        This method processes each detected level sequentially, using parent codes
        from previous levels to scope the search space and improve accuracy.
        
        Args:
            entities_df: Entity DataFrame with hierarchical name columns
            lgd_df: LGD reference DataFrame with hierarchical codes
            
        Returns:
            Tuple of (updated entities DataFrame, mapping statistics dict)
        """
        self.logger.info("Starting hierarchical code mapping for all levels")
        
        # Get detected levels in hierarchical order
        detected_levels = self.hierarchy_config.get_detected_levels_in_order()
        
        # Track overall statistics
        all_stats = {}
        updated_df = entities_df.copy()
        
        # Process each level in order
        for level in detected_levels:
            # Check if code mapping is enabled for this level
            if not self.hierarchy_config.is_code_mapping_enabled(level.name):
                self.logger.info(f"Skipping code mapping for level '{level.name}' (disabled in config)")
                continue
            
            # Check if code column already exists and has values
            if level.code_column in updated_df.columns:
                existing_codes = updated_df[level.code_column].notna().sum()
                total_records = len(updated_df)
                
                if existing_codes == total_records:
                    self.logger.info(
                        f"Skipping code mapping for level '{level.name}' "
                        f"(all {total_records} records already have codes)"
                    )
                    all_stats[level.name] = {
                        'total_unique': updated_df[level.name_column].nunique(),
                        'successfully_mapped': existing_codes,
                        'failed_to_map': 0,
                        'success_rate': 100.0,
                        'exact_matches': existing_codes,
                        'fuzzy_matches': 0,
                        'skipped': True
                    }
                    continue
                elif existing_codes > 0:
                    self.logger.info(
                        f"Level '{level.name}' has {existing_codes}/{total_records} "
                        f"existing codes, will map remaining {total_records - existing_codes}"
                    )
            
            # Get parent level for scoping
            parent_level = self.hierarchy_config.get_parent_level(level.name)
            parent_code_column = parent_level.code_column if parent_level else None
            
            # Map codes for this level
            self.logger.info(f"Mapping codes for level '{level.name}'...")
            updated_df, level_stats = self.map_level(
                level.name,
                updated_df,
                lgd_df,
                parent_code_column
            )
            
            # Store statistics
            all_stats[level.name] = level_stats
            
            # Log level summary with hierarchical context
            self.logger.info(
                f"Level '{level.name}' mapping complete: "
                f"{level_stats['successfully_mapped']}/{level_stats['total_unique']} "
                f"unique names mapped ({level_stats['success_rate']:.1f}% success rate, "
                f"{level_stats['exact_matches']} exact, {level_stats['fuzzy_matches']} fuzzy)",
                extra={
                    'hierarchy_level': level.name,
                    'total_unique': level_stats['total_unique'],
                    'successfully_mapped': level_stats['successfully_mapped'],
                    'failed_to_map': level_stats['failed_to_map'],
                    'success_rate': level_stats['success_rate'],
                    'exact_matches': level_stats['exact_matches'],
                    'fuzzy_matches': level_stats['fuzzy_matches'],
                    'parent_level': parent_level.name if parent_level else None
                }
            )
        
        # Log overall summary with hierarchical context
        total_levels_mapped = len([s for s in all_stats.values() if not s.get('skipped', False)])
        total_mapped = sum(s['successfully_mapped'] for s in all_stats.values())
        total_failed = sum(s['failed_to_map'] for s in all_stats.values())
        
        self.logger.info(
            f"Hierarchical code mapping completed for {total_levels_mapped} levels: "
            f"{total_mapped} successfully mapped, {total_failed} failed",
            extra={
                'levels_mapped': total_levels_mapped,
                'total_successfully_mapped': total_mapped,
                'total_failed': total_failed,
                'level_statistics': all_stats
            }
        )
        
        # Check for critical failures and raise error if needed
        critical_failures = []
        for level_name, stats in all_stats.items():
            if stats.get('skipped', False):
                continue
            
            # Consider it a critical failure if success rate is below 50%
            if stats['success_rate'] < 50.0 and stats['total_unique'] > 10:
                critical_failures.append(level_name)
                self.logger.error(
                    f"Critical mapping failure at level '{level_name}': "
                    f"only {stats['success_rate']:.1f}% success rate",
                    extra={
                        'hierarchy_level': level_name,
                        'success_rate': stats['success_rate'],
                        'failed_count': stats['failed_to_map'],
                        'total_count': stats['total_unique']
                    }
                )
        
        # If there are critical failures, log warning but allow graceful degradation
        if critical_failures:
            self.logger.warning(
                f"Hierarchical mapping completed with critical failures at {len(critical_failures)} level(s): "
                f"{', '.join(critical_failures)}. Processing will continue with partial data.",
                extra={
                    'critical_failure_levels': critical_failures,
                    'graceful_degradation': True
                }
            )
        
        return updated_df, all_stats

    def map_level(
        self,
        level_name: str,
        entities_df: pd.DataFrame,
        lgd_df: pd.DataFrame,
        parent_code_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Map codes for a specific hierarchical level with parent scoping.
        
        This method maps names to codes for a single administrative level,
        using parent codes to scope the search and improve accuracy.
        
        Args:
            level_name: Name of level to map ('state', 'district', 'block', 'gp', 'village')
            entities_df: Entity DataFrame with name columns
            lgd_df: LGD reference DataFrame with codes
            parent_code_column: Column name of parent level code for scoping (None for top level)
            
        Returns:
            Tuple of (updated DataFrame, level statistics dict)
        """
        # Get level configuration
        level = self.hierarchy_config.get_level(level_name)
        if not level:
            raise ValueError(f"Level '{level_name}' not found in hierarchy configuration")
        
        self.logger.debug(
            f"Mapping level '{level_name}': name_column='{level.name_column}', "
            f"code_column='{level.code_column}', parent_column='{parent_code_column}'"
        )
        
        # Validate required columns exist
        if level.name_column not in entities_df.columns:
            raise ValidationError(
                message=f"Missing required column '{level.name_column}' in entities DataFrame",
                field_name=level.name_column,
                validation_rules=[f"entities DataFrame must contain '{level.name_column}' column"]
            )
        
        # Special case: LGD uses 'state_name' instead of 'state'
        lgd_name_column = level.name_column
        if level.name == 'state' and level.name_column not in lgd_df.columns and 'state_name' in lgd_df.columns:
            lgd_name_column = 'state_name'
        
        if lgd_name_column not in lgd_df.columns:
            raise ValidationError(
                message=f"Missing required column '{lgd_name_column}' in LGD DataFrame",
                field_name=lgd_name_column,
                validation_rules=[f"LGD DataFrame must contain '{lgd_name_column}' column"]
            )
        
        if level.code_column not in lgd_df.columns:
            raise ValidationError(
                message=f"Missing required column '{level.code_column}' in LGD DataFrame",
                field_name=level.code_column,
                validation_rules=[f"LGD DataFrame must contain '{level.code_column}' column"]
            )
        
        # Extract unique names that need mapping
        updated_df = entities_df.copy()
        
        # Initialize code column if it doesn't exist
        if level.code_column not in updated_df.columns:
            updated_df[level.code_column] = None
        
        # Get names that need mapping (where code is missing)
        needs_mapping_mask = updated_df[level.code_column].isna() & updated_df[level.name_column].notna()
        unique_names = updated_df.loc[needs_mapping_mask, level.name_column].unique()
        total_unique = len(unique_names)
        
        self.logger.info(f"Found {total_unique} unique {level_name} names to map")
        
        if total_unique == 0:
            return updated_df, {
                'total_unique': 0,
                'successfully_mapped': 0,
                'failed_to_map': 0,
                'success_rate': 100.0,
                'exact_matches': 0,
                'fuzzy_matches': 0,
                'unmapped_names': []
            }
        
        # Build lookup dictionary for this level
        lookup = self._build_level_lookup(lgd_df, level, parent_code_column)
        
        # Get fuzzy threshold for this level
        fuzzy_threshold = self.hierarchy_config.get_fuzzy_threshold(level_name) or 85
        
        # Track mapping results
        mapping_details = {}
        exact_matches = 0
        fuzzy_matches = 0
        unmapped_names = []
        
        # Map each unique name to a code
        # When parent scoping is used, we need to map each (name, parent) combination
        if parent_code_column and parent_code_column in updated_df.columns:
            # Build a mapping for each (name, parent_code) combination
            name_parent_mapping = {}
            
            for name in unique_names:
                # Extract alternative names (including parenthetical text)
                alternative_names = self._extract_alternative_names(name)
                
                # Get all parent codes for entities with this name
                entity_rows = updated_df[
                    (updated_df[level.name_column] == name) & 
                    needs_mapping_mask
                ]
                
                parent_codes = entity_rows[parent_code_column].dropna().unique()
                
                # Map each (name, parent_code) combination
                for parent_code in parent_codes:
                    code = None
                    match_type = None
                    
                    # Try exact matching with parent context
                    for alt_name in alternative_names:
                        normalized_name = self._normalize_name(alt_name)
                        code = self._find_code_exact(normalized_name, parent_code, lookup)
                        if code:
                            match_type = 'exact'
                            self.logger.debug(
                                f"Exact match: '{name}' (using '{alt_name}') "
                                f"with parent {parent_code} -> code {code}"
                            )
                            break
                    
                    # If no exact match, try fuzzy matching
                    if not code:
                        for alt_name in alternative_names:
                            normalized_name = self._normalize_name(alt_name)
                            code = self._find_code_fuzzy(
                                normalized_name, 
                                parent_code, 
                                lookup, 
                                fuzzy_threshold
                            )
                            if code:
                                match_type = 'fuzzy'
                                self.logger.debug(
                                    f"Fuzzy match: '{name}' (using '{alt_name}') "
                                    f"with parent {parent_code} -> code {code}"
                                )
                                break
                    
                    # Track results
                    if code:
                        parent_code_str = str(int(parent_code)) if isinstance(parent_code, float) else str(parent_code)
                        name_parent_mapping[(name, parent_code_str)] = code
                        
                        if match_type == 'exact':
                            exact_matches += 1
                        else:
                            fuzzy_matches += 1
                    else:
                        self.logger.warning(
                            f"Could not map {level_name} name: '{name}' with parent {parent_code}"
                        )
            
            # Apply mappings based on (name, parent_code) combinations
            for idx in updated_df[needs_mapping_mask].index:
                name = updated_df.loc[idx, level.name_column]
                parent_code = updated_df.loc[idx, parent_code_column]
                
                if pd.notna(parent_code):
                    parent_code_str = str(int(parent_code)) if isinstance(parent_code, float) else str(parent_code)
                    key = (name, parent_code_str)
                    if key in name_parent_mapping:
                        updated_df.at[idx, level.code_column] = name_parent_mapping[key]
            
            # Count successfully mapped unique names
            successfully_mapped = len(set(name for name, _ in name_parent_mapping.keys()))
            unmapped_names = [name for name in unique_names if not any(
                name == n for n, _ in name_parent_mapping.keys()
            )]
        else:
            # No parent scoping - map each unique name directly
            for name in unique_names:
                # Extract alternative names (including parenthetical text)
                alternative_names = self._extract_alternative_names(name)
                
                code = None
                match_type = None
                
                # Try exact matching
                for alt_name in alternative_names:
                    normalized_name = self._normalize_name(alt_name)
                    code = self._find_code_exact(normalized_name, None, lookup)
                    if code:
                        match_type = 'exact'
                        self.logger.debug(f"Exact match: '{name}' (using '{alt_name}') -> code {code}")
                        break
                
                # If no exact match, try fuzzy matching
                if not code:
                    for alt_name in alternative_names:
                        normalized_name = self._normalize_name(alt_name)
                        code = self._find_code_fuzzy(normalized_name, None, lookup, fuzzy_threshold)
                        if code:
                            match_type = 'fuzzy'
                            self.logger.debug(
                                f"Fuzzy match: '{name}' (using '{alt_name}') -> code {code}"
                            )
                            break
                
                # Track results
                if code:
                    if match_type == 'exact':
                        exact_matches += 1
                    else:
                        fuzzy_matches += 1
                    mapping_details[name] = code
                else:
                    unmapped_names.append(name)
                    self.logger.warning(f"Could not map {level_name} name: '{name}'")
        
        # Update DataFrame with mapped codes (if not using parent scoping)
        if not parent_code_column or parent_code_column not in updated_df.columns:
            updated_df.loc[needs_mapping_mask, level.code_column] = (
                updated_df.loc[needs_mapping_mask, level.name_column].map(mapping_details)
            )
            successfully_mapped = len(mapping_details)
        # else: already updated in the parent scoping block above
        
        # Calculate statistics
        failed_to_map = total_unique - successfully_mapped
        success_rate = (successfully_mapped / total_unique * 100) if total_unique > 0 else 0
        
        stats = {
            'total_unique': total_unique,
            'successfully_mapped': successfully_mapped,
            'failed_to_map': failed_to_map,
            'success_rate': success_rate,
            'exact_matches': exact_matches,
            'fuzzy_matches': fuzzy_matches,
            'unmapped_names': unmapped_names
        }
        
        if failed_to_map > 0:
            self.logger.warning(
                f"{failed_to_map} {level_name} names could not be mapped: "
                f"{', '.join(unmapped_names[:5])}{'...' if len(unmapped_names) > 5 else ''}",
                extra={
                    'hierarchy_level': level_name,
                    'failed_count': failed_to_map,
                    'unmapped_names': unmapped_names,
                    'parent_level': parent_code_column,
                    'fallback_available': False
                }
            )
        
        return updated_df, stats

    def _build_level_lookup(
        self,
        lgd_df: pd.DataFrame,
        level: HierarchyLevel,
        parent_code_column: Optional[str] = None
    ) -> Dict[Tuple, str]:
        """
        Build lookup dictionary for a hierarchical level with caching.
        
        Creates a dictionary mapping (parent_code, normalized_name) tuples to codes.
        If no parent_code_column is provided, maps normalized_name directly to code.
        Uses caching to avoid rebuilding lookups for the same level.
        
        Args:
            lgd_df: LGD reference DataFrame
            level: HierarchyLevel configuration
            parent_code_column: Optional parent code column for scoping
            
        Returns:
            Dictionary mapping lookup keys to codes
        """
        # Check cache first if caching is enabled
        cache_key = f"{level.name}_{parent_code_column}"
        if self.enable_caching and cache_key in self.level_mappers:
            self.logger.debug(f"Using cached lookup for level '{level.name}'")
            return self.level_mappers[cache_key]
        
        lookup = {}
        
        # Determine columns to use
        # Special case: LGD uses 'state_name' instead of 'state'
        name_column = level.name_column
        if level.name == 'state' and name_column not in lgd_df.columns and 'state_name' in lgd_df.columns:
            name_column = 'state_name'
        
        columns_to_extract = [name_column, level.code_column]
        if parent_code_column and parent_code_column in lgd_df.columns:
            columns_to_extract.append(parent_code_column)
            use_parent = True
        else:
            use_parent = False
        
        # Extract unique combinations
        level_data = lgd_df[columns_to_extract].drop_duplicates()
        
        for _, row in level_data.iterrows():
            name = row[name_column]
            code = row[level.code_column]
            
            if pd.notna(name) and pd.notna(code):
                normalized_name = self._normalize_name(name)
                
                # Convert code to string for consistency
                code_str = str(int(code)) if isinstance(code, float) else str(code)
                
                # Create lookup key
                if use_parent and parent_code_column in row.index:
                    parent_code = row[parent_code_column]
                    if pd.notna(parent_code):
                        parent_code_str = str(int(parent_code)) if isinstance(parent_code, float) else str(parent_code)
                        lookup_key = (parent_code_str, normalized_name)
                    else:
                        # Parent code is null, use name only
                        lookup_key = (None, normalized_name)
                else:
                    # No parent context
                    lookup_key = (None, normalized_name)
                
                # Handle duplicates
                if lookup_key in lookup:
                    # Multiple LGD records for same combination
                    self.logger.debug(
                        f"Multiple LGD records found for {level.name} '{name}' "
                        f"with parent {lookup_key[0]}, using first occurrence (code: {lookup[lookup_key]})"
                    )
                else:
                    lookup[lookup_key] = code_str
        
        self.logger.debug(
            f"Built {level.name} lookup with {len(lookup)} entries "
            f"(parent_scoped: {parent_code_column is not None})"
        )
        
        # Cache the lookup if caching is enabled
        if self.enable_caching:
            self.level_mappers[cache_key] = lookup
            self._update_cache_size()
        
        return lookup
    
    def _find_code_exact(
        self,
        normalized_name: str,
        parent_code: Optional[str],
        lookup: Dict[Tuple, str]
    ) -> Optional[str]:
        """
        Find code using exact normalized name match with parent context and caching.
        
        Args:
            normalized_name: Normalized name to search for
            parent_code: Optional parent code for scoping
            lookup: Lookup dictionary mapping (parent_code, name) to code
            
        Returns:
            Code if found, None otherwise
        """
        if not normalized_name:
            return None
        
        # Convert parent_code to string if needed
        if parent_code is not None:
            parent_code_str = str(int(parent_code)) if isinstance(parent_code, float) else str(parent_code)
        else:
            parent_code_str = None
        
        # Check cache first if enabled
        cache_key = f"exact_{parent_code_str}_{normalized_name}"
        if self.enable_caching and cache_key in self._lookup_cache:
            self._cache_hits += 1
            return self._lookup_cache[cache_key]
        
        self._cache_misses += 1
        
        # Try lookup with parent context
        lookup_key = (parent_code_str, normalized_name)
        code = lookup.get(lookup_key)
        
        if code:
            # Cache the result
            if self.enable_caching:
                self._lookup_cache[cache_key] = code
                self._update_cache_size()
            return code
        
        # If parent context didn't work, try without parent (fallback)
        if parent_code_str is not None:
            lookup_key = (None, normalized_name)
            code = lookup.get(lookup_key)
            if code:
                self.logger.debug(
                    f"Found match without parent context for '{normalized_name}' "
                    f"(parent {parent_code_str} not found)"
                )
                # Cache the result
                if self.enable_caching:
                    self._lookup_cache[cache_key] = code
                    self._update_cache_size()
                return code
        
        # Cache negative result to avoid repeated lookups
        if self.enable_caching:
            self._lookup_cache[cache_key] = None
            self._update_cache_size()
        
        return None
    
    def _find_code_fuzzy(
        self,
        normalized_name: str,
        parent_code: Optional[str],
        lookup: Dict[Tuple, str],
        threshold: int
    ) -> Optional[str]:
        """
        Find code using fuzzy string matching with parent context and caching.
        
        Args:
            normalized_name: Normalized name to search for
            parent_code: Optional parent code for scoping
            lookup: Lookup dictionary mapping (parent_code, name) to code
            threshold: Minimum similarity score (0-100) for a match
            
        Returns:
            Code if fuzzy match found above threshold, None otherwise
        """
        if not normalized_name:
            return None
        
        # Convert parent_code to string if needed
        if parent_code is not None:
            parent_code_str = str(int(parent_code)) if isinstance(parent_code, float) else str(parent_code)
        else:
            parent_code_str = None
        
        # Check cache first if enabled
        cache_key = f"fuzzy_{threshold}_{parent_code_str}_{normalized_name}"
        if self.enable_caching and cache_key in self._lookup_cache:
            self._cache_hits += 1
            return self._lookup_cache[cache_key]
        
        self._cache_misses += 1
        
        best_match = None
        best_score = 0
        
        # Search through lookup entries
        for (lookup_parent, lookup_name), code in lookup.items():
            # If we have parent context, only compare within same parent
            if parent_code_str is not None and lookup_parent is not None:
                if lookup_parent != parent_code_str:
                    continue
            
            # Calculate similarity score
            score = fuzz.ratio(normalized_name, lookup_name)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = code
                self.logger.debug(
                    f"Fuzzy match candidate: '{normalized_name}' ~ '{lookup_name}' "
                    f"(parent: {parent_code_str}, score: {score})"
                )
        
        # If no match with parent context, try without parent (fallback)
        if not best_match and parent_code_str is not None:
            for (lookup_parent, lookup_name), code in lookup.items():
                # Skip entries that have a different parent
                if lookup_parent is not None and lookup_parent != parent_code_str:
                    continue
                
                score = fuzz.ratio(normalized_name, lookup_name)
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = code
                    self.logger.debug(
                        f"Fuzzy match candidate (no parent filter): '{normalized_name}' ~ '{lookup_name}' "
                        f"(score: {score})"
                    )
        
        # Cache the result
        if self.enable_caching:
            self._lookup_cache[cache_key] = best_match
            self._update_cache_size()
        
        return best_match
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize administrative name for comparison.
        
        Removes parenthetical text, trims whitespace, converts to lowercase.
        
        Args:
            name: Original name
            
        Returns:
            Normalized name
        """
        if pd.isna(name) or not isinstance(name, str):
            return ""
        
        # Remove text in parentheses (e.g., "Keonjhar (Kendujhar)" -> "Keonjhar")
        normalized = re.sub(r'\s*\([^)]*\)', '', name)
        
        # Trim whitespace and convert to lowercase
        normalized = normalized.strip().lower()
        
        return normalized
    
    def _extract_alternative_names(self, name: str) -> List[str]:
        """
        Extract alternative names from parenthetical text.
        
        For names like "Keonjhar (Kendujhar)", returns both "keonjhar" and "kendujhar".
        This helps match against different naming conventions in the LGD data.
        
        Args:
            name: Original name
            
        Returns:
            List of normalized alternative names
        """
        if pd.isna(name) or not isinstance(name, str):
            return []
        
        alternatives = []
        
        # Extract main name (without parentheses)
        main_name = re.sub(r'\s*\([^)]*\)', '', name).strip().lower()
        if main_name:
            alternatives.append(main_name)
        
        # Extract text inside parentheses
        parenthetical_matches = re.findall(r'\(([^)]+)\)', name)
        for match in parenthetical_matches:
            alt_name = match.strip().lower()
            if alt_name and alt_name not in alternatives:
                alternatives.append(alt_name)
        
        return alternatives

    def _update_cache_size(self):
        """
        Update cache size tracking and enforce size limits.
        
        Estimates cache size and clears cache if it exceeds the limit.
        """
        if not self.enable_caching:
            return
        
        # Estimate cache size (rough approximation)
        # Each cache entry is approximately 100 bytes (key + value + overhead)
        estimated_size_bytes = (
            len(self._lookup_cache) * 100 +
            sum(len(mapper) * 150 for mapper in self.level_mappers.values())
        )
        
        self._cache_size_bytes = estimated_size_bytes
        cache_size_mb = estimated_size_bytes / (1024 * 1024)
        
        # Check if cache exceeds limit
        if cache_size_mb > self.cache_size_limit_mb:
            self.logger.warning(
                f"Cache size ({cache_size_mb:.2f} MB) exceeds limit ({self.cache_size_limit_mb} MB), "
                f"clearing lookup cache"
            )
            self._clear_lookup_cache()
    
    def _clear_lookup_cache(self):
        """
        Clear the lookup cache to free memory.
        
        Keeps level_mappers cache but clears individual lookup results.
        """
        self._lookup_cache.clear()
        self._cache_size_bytes = sum(len(mapper) * 150 for mapper in self.level_mappers.values())
        self.logger.debug("Cleared lookup cache")
    
    def clear_all_caches(self):
        """
        Clear all caches including level mappers.
        
        This should be called when switching to a new dataset or when
        memory needs to be freed.
        """
        self._lookup_cache.clear()
        self.level_mappers.clear()
        self._cache_size_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.info("Cleared all hierarchical code mapper caches")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_lookups = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_lookups * 100) if total_lookups > 0 else 0
        
        return {
            'cache_enabled': self.enable_caching,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': hit_rate,
            'cache_size_mb': self._cache_size_bytes / (1024 * 1024),
            'cache_size_limit_mb': self.cache_size_limit_mb,
            'lookup_cache_entries': len(self._lookup_cache),
            'level_mapper_entries': sum(len(mapper) for mapper in self.level_mappers.values())
        }
