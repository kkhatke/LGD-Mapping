"""
Mapping orchestration engine for LGD mapping application.

This module provides the MappingEngine class that coordinates all matching strategies
in a sequential pipeline and aggregates results with comprehensive statistics.
"""

import pandas as pd
import time
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm

from .models import EntityRecord, LGDRecord, MappingResult
from .matching.exact_matcher import ExactMatcher
from .matching.fuzzy_matcher import FuzzyMatcher
from .data_loader import DataLoader
from .config import MappingConfig, ProcessingStats
from .logging_config import MappingLogger
from .exceptions import MappingProcessError, DataQualityError
from .utils.data_validator import DataValidator, DataQualityReport
from .utils.error_handler import ErrorHandler, create_error_context, log_error_details
from .utils.district_mapper import DistrictCodeMapper


class MappingEngine:
    """
    Orchestrates the complete mapping process using multiple matching strategies.
    
    This class coordinates exact matching followed by fuzzy matching with multiple
    thresholds, providing comprehensive progress tracking and statistics.
    """
    
    def __init__(self, config: MappingConfig, logger: Optional[MappingLogger] = None):
        """
        Initialize the MappingEngine.
        
        Args:
            config: Configuration object with mapping parameters
            logger: Optional logger instance for logging operations
        """
        self.config = config
        self.logger = logger or MappingLogger()
        
        # Initialize error handling
        self.error_handler = ErrorHandler(self.logger.logger)
        
        # Initialize components with error handling
        self.data_loader = DataLoader(
            logger=self.logger.logger,
            error_handler=self.error_handler
        )
        self.exact_matcher = ExactMatcher(
            logger=self.logger.logger,
            error_handler=self.error_handler
        )
        
        # Initialize data validator
        self.data_validator = DataValidator(self.logger.logger)
        
        # Initialize fuzzy matchers for each threshold
        self.fuzzy_matchers = {}
        for threshold in self.config.fuzzy_thresholds:
            self.fuzzy_matchers[threshold] = FuzzyMatcher(threshold, self.logger.logger)
        
        # Results storage
        self.all_results: List[MappingResult] = []
        self.processing_stats = ProcessingStats()
        
        # Data storage
        self.entities_df: Optional[pd.DataFrame] = None
        self.lgd_df: Optional[pd.DataFrame] = None
        
        # Validation results storage
        self.validation_results: Optional[Dict[str, Any]] = None
    
    def run_complete_mapping(self) -> Tuple[List[MappingResult], ProcessingStats]:
        """
        Run the complete mapping pipeline from data loading to final results.
        
        Returns:
            Tuple of (all mapping results, processing statistics)
        """
        start_time = time.time()
        
        try:
            # Load data with performance monitoring
            self.logger.log_phase_start("Data Loading")
            load_start = time.time()
            self._load_data()
            load_duration = time.time() - load_start
            self.logger.log_phase_complete("Data Loading", 
                                         len(self.entities_df) + len(self.lgd_df), 
                                         load_duration)
            
            # Log processing start
            self.logger.log_processing_start(len(self.entities_df), len(self.lgd_df))
            
            # Initialize processing stats
            self.processing_stats.total_entities = len(self.entities_df)
            
            # Check if we need chunk processing for large datasets
            if self.config.chunk_size and len(self.entities_df) > self.config.chunk_size:
                self.logger.info(f"Large dataset detected. Using chunk processing with size: {self.config.chunk_size:,}")
                self._run_chunked_mapping_pipeline()
            else:
                # Run standard mapping pipeline
                self._run_mapping_pipeline()
            
            # Calculate final statistics
            self.processing_stats.processing_time = time.time() - start_time
            self._calculate_final_statistics()
            
            # Log completion
            self.logger.log_processing_complete(self.processing_stats)
            
            return self.all_results, self.processing_stats
            
        except Exception as e:
            self.logger.error(f"Error in complete mapping process: {e}")
            raise
    
    def _load_data(self):
        """Load entities and LGD codes data with comprehensive validation."""
        try:
            # Load entities
            self.entities_df = self.data_loader.load_entities(self.config.input_entities_file)
            self.logger.info(f"Loaded {len(self.entities_df)} entity records")
            
            # Load LGD codes
            self.lgd_df = self.data_loader.load_lgd_codes(self.config.input_codes_file)
            self.logger.info(f"Loaded {len(self.lgd_df)} LGD code records")
            
            # Perform district code mapping using DistrictCodeMapper (if enabled)
            if self.config.enable_district_code_mapping:
                self.logger.info("Starting district code mapping")
                district_mapper = DistrictCodeMapper(self.logger.logger)
                
                # Preserve existing district_code values by creating a backup
                existing_codes = self.entities_df['district_code'].copy() if 'district_code' in self.entities_df.columns else None
                
                # Call map_district_codes to enrich entities with district codes
                self.entities_df, mapping_stats = district_mapper.map_district_codes(
                    self.entities_df,
                    self.lgd_df,
                    fuzzy_threshold=self.config.district_fuzzy_threshold
                )
                
                # Restore existing district codes where they were already present
                if existing_codes is not None:
                    # Only overwrite NaN values, preserve existing codes
                    mask = existing_codes.notna()
                    self.entities_df.loc[mask, 'district_code'] = existing_codes[mask]
                    preserved_count = mask.sum()
                    if preserved_count > 0:
                        self.logger.info(f"Preserved {preserved_count} existing district codes")
                
                # Log mapping statistics
                self.logger.info(
                    f"District code mapping completed: {mapping_stats['successfully_mapped']}/{mapping_stats['total_unique_districts']} "
                    f"districts mapped ({mapping_stats['success_rate']:.1f}% success rate)"
                )
                self.logger.info(
                    f"Mapping breakdown: {mapping_stats['exact_matches']} exact matches, "
                    f"{mapping_stats['fuzzy_matches']} fuzzy matches"
                )
                
                # Add warning log if success rate is below 50%
                if mapping_stats['success_rate'] < 50:
                    self.logger.warning(
                        f"Low district code mapping success rate: {mapping_stats['success_rate']:.1f}%. "
                        f"This may impact matching performance."
                    )
                    if mapping_stats['unmapped_districts']:
                        self.logger.warning(
                            f"Unmapped districts: {', '.join(mapping_stats['unmapped_districts'][:10])}"
                            + (f" and {len(mapping_stats['unmapped_districts']) - 10} more" 
                               if len(mapping_stats['unmapped_districts']) > 10 else "")
                        )
            else:
                self.logger.info("District code mapping is disabled in configuration")
            
            # Perform comprehensive data validation
            self.logger.info("Starting comprehensive data validation")
            self.validation_results = self.data_loader.perform_comprehensive_validation(
                self.entities_df, self.lgd_df
            )
            
            # Log validation summary
            self.logger.info("Data validation completed")
            self.logger.info(f"Overall data quality score: {self.validation_results['overall_quality_score']:.1f}/100")
            
            # Handle critical validation issues
            if self.validation_results['critical_issues_found']:
                self.logger.error("Critical data quality issues detected")
                
                # Log detailed validation summary
                validation_summary = self.validation_results.get('validation_summary', '')
                if validation_summary:
                    self.logger.error(f"Validation Summary:\n{validation_summary}")
                
                # Decide whether to continue or abort based on quality score
                if self.validation_results['overall_quality_score'] < 30:
                    raise DataQualityError(
                        f"Data quality is too low to proceed with mapping: {self.validation_results['overall_quality_score']:.1f}/100",
                        quality_issue="critically_low_overall_quality",
                        affected_records=len(self.entities_df),
                        severity="critical",
                        recommendations=["Review and improve data quality before proceeding"]
                    )
                else:
                    self.logger.warning("Proceeding with mapping despite data quality issues")
            
        except (DataQualityError, MappingProcessError):
            raise
        except Exception as e:
            context = create_error_context(
                operation="load_data",
                entities_file=self.config.input_entities_file,
                lgd_file=self.config.input_codes_file
            )
            log_error_details(self.logger.logger, e, context)
            
            raise MappingProcessError(
                f"Failed to load and validate data: {str(e)}",
                strategy="data_loading",
                original_error=e
            )
    

    
    def _run_mapping_pipeline(self):
        """Run the sequential mapping pipeline."""
        # Start with all entities
        remaining_entities = self.entities_df.copy()
        
        # Phase 1: Exact matching
        self.logger.log_phase_start("Exact Matching")
        exact_start = time.time()
        exact_results = self.exact_matcher.match(remaining_entities, self.lgd_df)
        exact_duration = time.time() - exact_start
        
        # Separate matched and unmatched from exact matching
        exact_matched = [r for r in exact_results if r.is_matched()]
        exact_unmatched_indices = [i for i, r in enumerate(exact_results) if not r.is_matched()]
        
        self.processing_stats.exact_matches = len(exact_matched)
        self.all_results.extend(exact_matched)
        
        self.logger.log_match_statistics("Exact Matching", 
                                       len(exact_matched), 
                                       len(exact_results), 
                                       exact_duration)
        
        # Get unmatched entities for fuzzy matching
        if exact_unmatched_indices:
            remaining_entities = remaining_entities.iloc[exact_unmatched_indices].copy()
        else:
            remaining_entities = pd.DataFrame()
        
        # Phase 2: Fuzzy matching with multiple thresholds
        for threshold in self.config.fuzzy_thresholds:
            if remaining_entities.empty:
                self.logger.info(f"No remaining entities for fuzzy matching at {threshold}%")
                self.processing_stats.fuzzy_matches[threshold] = 0
                continue
            
            self.logger.log_phase_start(f"Fuzzy Matching ({threshold}%)")
            fuzzy_start = time.time()
            
            fuzzy_matcher = self.fuzzy_matchers[threshold]
            fuzzy_results = fuzzy_matcher.match(remaining_entities, self.lgd_df)
            fuzzy_duration = time.time() - fuzzy_start
            
            # Separate matched and unmatched from fuzzy matching
            fuzzy_matched = [r for r in fuzzy_results if r.is_matched()]
            fuzzy_unmatched_indices = [i for i, r in enumerate(fuzzy_results) if not r.is_matched()]
            
            self.processing_stats.fuzzy_matches[threshold] = len(fuzzy_matched)
            self.all_results.extend(fuzzy_matched)
            
            self.logger.log_match_statistics(f"Fuzzy Matching ({threshold}%)", 
                                           len(fuzzy_matched), 
                                           len(fuzzy_results), 
                                           fuzzy_duration)
            
            # Update remaining entities for next threshold
            if fuzzy_unmatched_indices:
                remaining_entities = remaining_entities.iloc[fuzzy_unmatched_indices].copy()
                # Also collect unmatched results for final output
                fuzzy_unmatched = [fuzzy_results[i] for i in fuzzy_unmatched_indices]
                # Only add unmatched results from the last threshold to avoid duplicates
                if threshold == self.config.fuzzy_thresholds[-1]:
                    self.all_results.extend(fuzzy_unmatched)
            else:
                remaining_entities = pd.DataFrame()
        
        # Calculate unmatched count
        self.processing_stats.unmatched = len([r for r in self.all_results if not r.is_matched()])
    
    def _run_chunked_mapping_pipeline(self):
        """Run mapping pipeline with chunk processing for large datasets."""
        import math
        
        chunk_size = self.config.chunk_size
        total_entities = len(self.entities_df)
        num_chunks = math.ceil(total_entities / chunk_size)
        
        self.logger.info(f"Processing {total_entities:,} entities in {num_chunks} chunks of {chunk_size:,}")
        
        # Process entities in chunks
        with tqdm(total=total_entities, desc="Processing entities", unit="entities") as pbar:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, total_entities)
                
                chunk_entities = self.entities_df.iloc[start_idx:end_idx].copy()
                self.logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}: entities {start_idx:,}-{end_idx:,}")
                
                # Process this chunk through the mapping pipeline
                chunk_results = self._process_entity_chunk(chunk_entities)
                self.all_results.extend(chunk_results)
                
                # Update progress bar
                pbar.update(len(chunk_entities))
                
                # Force garbage collection every few chunks for memory management
                if (chunk_idx + 1) % 5 == 0:
                    import gc
                    gc.collect()
                    self.logger.info(f"Completed {chunk_idx + 1}/{num_chunks} chunks, performed garbage collection")
        
        # Calculate statistics from all results
        self._calculate_chunked_statistics()
    
    def _process_entity_chunk(self, chunk_entities: pd.DataFrame) -> List[MappingResult]:
        """Process a single chunk of entities through the mapping pipeline."""
        chunk_results = []
        remaining_entities = chunk_entities.copy()
        
        # Phase 1: Exact matching for this chunk
        exact_results = self.exact_matcher.match(remaining_entities, self.lgd_df)
        exact_matched = [r for r in exact_results if r.is_matched()]
        exact_unmatched_indices = [i for i, r in enumerate(exact_results) if not r.is_matched()]
        
        chunk_results.extend(exact_matched)
        self.processing_stats.exact_matches += len(exact_matched)
        
        # Get unmatched entities for fuzzy matching
        if exact_unmatched_indices:
            remaining_entities = remaining_entities.iloc[exact_unmatched_indices].copy()
        else:
            remaining_entities = pd.DataFrame()
        
        # Phase 2: Fuzzy matching for this chunk
        for threshold in self.config.fuzzy_thresholds:
            if remaining_entities.empty:
                break
            
            fuzzy_matcher = self.fuzzy_matchers[threshold]
            fuzzy_results = fuzzy_matcher.match(remaining_entities, self.lgd_df)
            
            fuzzy_matched = [r for r in fuzzy_results if r.is_matched()]
            fuzzy_unmatched_indices = [i for i, r in enumerate(fuzzy_results) if not r.is_matched()]
            
            chunk_results.extend(fuzzy_matched)
            
            if threshold not in self.processing_stats.fuzzy_matches:
                self.processing_stats.fuzzy_matches[threshold] = 0
            self.processing_stats.fuzzy_matches[threshold] += len(fuzzy_matched)
            
            # Update remaining entities for next threshold
            if fuzzy_unmatched_indices:
                remaining_entities = remaining_entities.iloc[fuzzy_unmatched_indices].copy()
                # Add unmatched results from the last threshold
                if threshold == self.config.fuzzy_thresholds[-1]:
                    fuzzy_unmatched = [fuzzy_results[i] for i in fuzzy_unmatched_indices]
                    chunk_results.extend(fuzzy_unmatched)
            else:
                remaining_entities = pd.DataFrame()
        
        return chunk_results
    
    def _calculate_chunked_statistics(self):
        """Calculate statistics after chunked processing."""
        # Count unmatched results
        unmatched_count = len([r for r in self.all_results if not r.is_matched()])
        self.processing_stats.unmatched = unmatched_count
        
        # Verify total count
        total_results = len(self.all_results)
        if total_results != self.processing_stats.total_entities:
            self.logger.warning(f"Result count mismatch after chunked processing: {total_results} results vs "
                              f"{self.processing_stats.total_entities} entities")
        
        self.logger.info(f"Chunked processing completed: {total_results:,} results processed")
    
    def _calculate_final_statistics(self):
        """Calculate final processing statistics."""
        # Verify total count
        total_results = len(self.all_results)
        if total_results != self.processing_stats.total_entities:
            self.logger.warning(f"Result count mismatch: {total_results} results vs "
                              f"{self.processing_stats.total_entities} entities")
        
        # Log detailed statistics
        self.logger.info("Final Mapping Statistics:")
        self.logger.info(f"  Total entities: {self.processing_stats.total_entities:,}")
        
        # Calculate rates safely to avoid division by zero
        if self.processing_stats.total_entities > 0:
            exact_rate = (self.processing_stats.exact_matches / self.processing_stats.total_entities) * 100
            unmatched_rate = (self.processing_stats.unmatched / self.processing_stats.total_entities) * 100
        else:
            exact_rate = 0.0
            unmatched_rate = 0.0
        
        self.logger.info(f"  Exact matches: {self.processing_stats.exact_matches:,} ({exact_rate:.2f}%)")
        
        for threshold, count in self.processing_stats.fuzzy_matches.items():
            rate = (count / self.processing_stats.total_entities * 100) if self.processing_stats.total_entities > 0 else 0
            self.logger.info(f"  Fuzzy matches ({threshold}%): {count:,} ({rate:.2f}%)")
        
        self.logger.info(f"  Unmatched: {self.processing_stats.unmatched:,} ({unmatched_rate:.2f}%)")
        self.logger.info(f"  Overall match rate: {self.processing_stats.get_match_rate():.2f}%")
    
    def get_results_by_type(self, match_type: str) -> List[MappingResult]:
        """
        Get results filtered by match type.
        
        Args:
            match_type: Type of match to filter by ('exact', 'fuzzy_95', 'fuzzy_90', 'unmatched')
            
        Returns:
            List of MappingResult objects of the specified type
        """
        return [r for r in self.all_results if r.match_type == match_type]
    
    def get_unmatched_with_alternatives(self) -> List[MappingResult]:
        """
        Get unmatched results that have alternative match suggestions.
        
        Returns:
            List of unmatched MappingResult objects with alternatives
        """
        return [r for r in self.all_results 
                if not r.is_matched() and r.alternative_matches]
    
    def get_unmatched_without_alternatives(self) -> List[MappingResult]:
        """
        Get unmatched results that have no alternative match suggestions.
        
        Returns:
            List of unmatched MappingResult objects without alternatives
        """
        return [r for r in self.all_results 
                if not r.is_matched() and not r.alternative_matches]
    
    def get_matching_quality_metrics(self) -> Dict[str, float]:
        """
        Calculate detailed matching quality metrics.
        
        Returns:
            Dictionary with various quality metrics
        """
        if not self.all_results:
            return {}
        
        total = len(self.all_results)
        matched = len([r for r in self.all_results if r.is_matched()])
        unmatched_with_alternatives = len(self.get_unmatched_with_alternatives())
        unmatched_without_alternatives = len(self.get_unmatched_without_alternatives())
        
        # Calculate confidence distribution
        high_confidence = len([r for r in self.all_results 
                              if r.is_matched() and r.get_confidence_level() == 'High'])
        medium_confidence = len([r for r in self.all_results 
                                if r.is_matched() and r.get_confidence_level() == 'Medium'])
        low_confidence = len([r for r in self.all_results 
                             if r.is_matched() and r.get_confidence_level() == 'Low'])
        
        return {
            'total_entities': total,
            'overall_match_rate': (matched / total) * 100,
            'exact_match_rate': (self.processing_stats.exact_matches / total) * 100,
            'fuzzy_match_rate': (sum(self.processing_stats.fuzzy_matches.values()) / total) * 100,
            'unmatched_rate': (self.processing_stats.unmatched / total) * 100,
            'unmatched_with_alternatives_rate': (unmatched_with_alternatives / total) * 100,
            'unmatched_without_alternatives_rate': (unmatched_without_alternatives / total) * 100,
            'high_confidence_rate': (high_confidence / total) * 100,
            'medium_confidence_rate': (medium_confidence / total) * 100,
            'low_confidence_rate': (low_confidence / total) * 100,
            'processing_time_seconds': self.processing_stats.processing_time
        }
    
    def get_validation_results(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive data validation results.
        
        Returns:
            Dictionary with validation results or None if validation not performed
        """
        return self.validation_results
    
    def get_data_quality_summary(self) -> str:
        """
        Get human-readable data quality summary.
        
        Returns:
            Data quality summary string
        """
        if not self.validation_results:
            return "Data validation not performed"
        
        return self.validation_results.get('validation_summary', 'No validation summary available')
    
    def get_data_quality_score(self) -> float:
        """
        Get overall data quality score.
        
        Returns:
            Quality score (0-100) or 0.0 if validation not performed
        """
        if not self.validation_results:
            return 0.0
        
        return self.validation_results.get('overall_quality_score', 0.0)
    
    def has_critical_quality_issues(self) -> bool:
        """
        Check if critical data quality issues were found.
        
        Returns:
            True if critical issues found, False otherwise
        """
        if not self.validation_results:
            return False
        
        return self.validation_results.get('critical_issues_found', False)
    
    def get_anomalies_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected anomalies.
        
        Returns:
            Dictionary with anomaly information
        """
        if not self.validation_results:
            return {'entity_anomalies': [], 'lgd_anomalies': [], 'total_anomalies': 0}
        
        entity_anomalies = self.validation_results.get('entity_anomalies', [])
        lgd_anomalies = self.validation_results.get('lgd_anomalies', [])
        
        return {
            'entity_anomalies': entity_anomalies,
            'lgd_anomalies': lgd_anomalies,
            'total_anomalies': len(entity_anomalies) + len(lgd_anomalies),
            'high_severity_anomalies': len([
                a for a in entity_anomalies + lgd_anomalies 
                if a.get('severity') == 'high'
            ])
        }
    
    def clear_results(self):
        """Clear all results and reset statistics."""
        self.all_results.clear()
        self.processing_stats = ProcessingStats()
        self.validation_results = None
        
        # Clear matcher caches
        self.exact_matcher.clear_cache()
        for matcher in self.fuzzy_matchers.values():
            matcher.clear_cache()
        
        # Reset error handler
        self.error_handler.reset_error_counts()
        
        self.logger.info("Cleared all results, caches, and validation data")


class ResultsAggregator:
    """
    Aggregates and analyzes mapping results from multiple strategies.
    
    This class provides functionality to combine results from different matching
    strategies and calculate comprehensive statistics and quality metrics.
    """
    
    def __init__(self, logger: Optional[MappingLogger] = None):
        """
        Initialize the ResultsAggregator.
        
        Args:
            logger: Optional logger instance for logging operations
        """
        self.logger = logger or MappingLogger()
    
    def aggregate_results(self, results_by_strategy: Dict[str, List[MappingResult]]) -> List[MappingResult]:
        """
        Aggregate results from multiple matching strategies, prioritizing higher confidence matches.
        
        Args:
            results_by_strategy: Dictionary mapping strategy names to their results
            
        Returns:
            List of aggregated MappingResult objects with best matches prioritized
        """
        self.logger.info("Aggregating results from multiple matching strategies")
        
        # Create a mapping from entity to best result
        entity_to_result = {}
        strategy_priority = ['exact', 'fuzzy_95', 'fuzzy_90', 'unmatched']
        
        # Process strategies in priority order
        for strategy in strategy_priority:
            if strategy not in results_by_strategy:
                continue
            
            results = results_by_strategy[strategy]
            self.logger.info(f"Processing {len(results)} results from {strategy} strategy")
            
            for result in results:
                # Create a unique key for the entity
                entity_key = self._create_entity_key(result.entity)
                
                # Only add if we don't have a result for this entity yet (higher priority wins)
                if entity_key not in entity_to_result:
                    entity_to_result[entity_key] = result
        
        aggregated_results = list(entity_to_result.values())
        self.logger.info(f"Aggregated {len(aggregated_results)} unique entity results")
        
        return aggregated_results
    
    def calculate_strategy_statistics(self, results: List[MappingResult]) -> Dict[str, Dict[str, int]]:
        """
        Calculate statistics for each matching strategy.
        
        Args:
            results: List of MappingResult objects
            
        Returns:
            Dictionary with statistics for each strategy
        """
        strategy_stats = {}
        
        # Group results by strategy
        for result in results:
            strategy = result.match_type
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'total': 0,
                    'matched': 0,
                    'unmatched': 0,
                    'with_alternatives': 0,
                    'without_alternatives': 0
                }
            
            strategy_stats[strategy]['total'] += 1
            
            if result.is_matched():
                strategy_stats[strategy]['matched'] += 1
            else:
                strategy_stats[strategy]['unmatched'] += 1
                if result.alternative_matches:
                    strategy_stats[strategy]['with_alternatives'] += 1
                else:
                    strategy_stats[strategy]['without_alternatives'] += 1
        
        # Log strategy statistics
        self.logger.info("Strategy-wise statistics:")
        for strategy, stats in strategy_stats.items():
            match_rate = (stats['matched'] / stats['total']) * 100 if stats['total'] > 0 else 0
            self.logger.info(f"  {strategy}: {stats['matched']}/{stats['total']} "
                           f"matched ({match_rate:.2f}%)")
        
        return strategy_stats
    
    def calculate_confidence_distribution(self, results: List[MappingResult]) -> Dict[str, int]:
        """
        Calculate distribution of confidence levels across all results.
        
        Args:
            results: List of MappingResult objects
            
        Returns:
            Dictionary with confidence level counts
        """
        confidence_counts = {'High': 0, 'Medium': 0, 'Low': 0, 'None': 0}
        
        for result in results:
            confidence = result.get_confidence_level()
            confidence_counts[confidence] += 1
        
        self.logger.info("Confidence distribution:")
        total = len(results)
        for confidence, count in confidence_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            self.logger.info(f"  {confidence}: {count} ({percentage:.2f}%)")
        
        return confidence_counts
    
    def calculate_match_score_statistics(self, results: List[MappingResult]) -> Dict[str, float]:
        """
        Calculate statistics for match scores.
        
        Args:
            results: List of MappingResult objects
            
        Returns:
            Dictionary with match score statistics
        """
        matched_results = [r for r in results if r.is_matched() and r.match_score is not None]
        
        if not matched_results:
            return {'count': 0, 'mean': 0.0, 'min': 0.0, 'max': 0.0}
        
        scores = [r.match_score for r in matched_results]
        
        stats = {
            'count': len(scores),
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'median': sorted(scores)[len(scores) // 2]
        }
        
        self.logger.info(f"Match score statistics: mean={stats['mean']:.3f}, "
                        f"min={stats['min']:.3f}, max={stats['max']:.3f}, "
                        f"median={stats['median']:.3f}")
        
        return stats
    
    def generate_quality_report(self, results: List[MappingResult]) -> Dict[str, any]:
        """
        Generate comprehensive quality report for mapping results.
        
        Args:
            results: List of MappingResult objects
            
        Returns:
            Dictionary with comprehensive quality metrics
        """
        self.logger.info("Generating comprehensive quality report")
        
        total_entities = len(results)
        matched_entities = len([r for r in results if r.is_matched()])
        unmatched_entities = total_entities - matched_entities
        
        # Calculate basic metrics
        overall_match_rate = (matched_entities / total_entities) * 100 if total_entities > 0 else 0
        
        # Get strategy statistics
        strategy_stats = self.calculate_strategy_statistics(results)
        
        # Get confidence distribution
        confidence_dist = self.calculate_confidence_distribution(results)
        
        # Get match score statistics
        score_stats = self.calculate_match_score_statistics(results)
        
        # Calculate alternative match statistics
        unmatched_with_alternatives = len([r for r in results 
                                         if not r.is_matched() and r.alternative_matches])
        unmatched_without_alternatives = unmatched_entities - unmatched_with_alternatives
        
        # Data quality indicators
        entities_with_district_codes = len([r for r in results 
                                          if r.entity.district_code is not None])
        
        quality_report = {
            'summary': {
                'total_entities': total_entities,
                'matched_entities': matched_entities,
                'unmatched_entities': unmatched_entities,
                'overall_match_rate': overall_match_rate,
                'entities_with_district_codes': entities_with_district_codes
            },
            'strategy_breakdown': strategy_stats,
            'confidence_distribution': confidence_dist,
            'match_score_statistics': score_stats,
            'unmatched_analysis': {
                'with_alternatives': unmatched_with_alternatives,
                'without_alternatives': unmatched_without_alternatives,
                'alternative_coverage_rate': (unmatched_with_alternatives / unmatched_entities) * 100 
                                           if unmatched_entities > 0 else 0
            },
            'data_quality_indicators': {
                'district_code_coverage': (entities_with_district_codes / total_entities) * 100 
                                        if total_entities > 0 else 0
            }
        }
        
        # Log summary
        self.logger.info("Quality Report Summary:")
        self.logger.info(f"  Overall match rate: {overall_match_rate:.2f}%")
        self.logger.info(f"  Unmatched with alternatives: {unmatched_with_alternatives}")
        self.logger.info(f"  Unmatched without alternatives: {unmatched_without_alternatives}")
        self.logger.info(f"  District code coverage: {quality_report['data_quality_indicators']['district_code_coverage']:.2f}%")
        
        return quality_report
    
    def identify_data_quality_issues(self, results: List[MappingResult]) -> List[Dict[str, any]]:
        """
        Identify potential data quality issues from mapping results.
        
        Args:
            results: List of MappingResult objects
            
        Returns:
            List of identified data quality issues
        """
        issues = []
        
        # Issue 1: High number of unmatched entities without alternatives
        unmatched_no_alternatives = len([r for r in results 
                                       if not r.is_matched() and not r.alternative_matches])
        if unmatched_no_alternatives > len(results) * 0.1:  # More than 10%
            issues.append({
                'type': 'high_unmatched_no_alternatives',
                'severity': 'medium',
                'count': unmatched_no_alternatives,
                'description': f'{unmatched_no_alternatives} entities have no matches or alternatives',
                'recommendation': 'Review entity data quality and LGD reference data completeness'
            })
        
        # Issue 2: Low district code coverage
        entities_with_codes = len([r for r in results if r.entity.district_code is not None])
        code_coverage = (entities_with_codes / len(results)) * 100 if results else 0
        if code_coverage < 50:  # Less than 50% coverage
            issues.append({
                'type': 'low_district_code_coverage',
                'severity': 'high',
                'coverage': code_coverage,
                'description': f'Only {code_coverage:.1f}% of entities have district codes',
                'recommendation': 'Improve district code mapping or data preprocessing'
            })
        
        # Issue 3: Low exact match rate
        exact_matches = len([r for r in results if r.match_type == 'exact'])
        exact_rate = (exact_matches / len(results)) * 100 if results else 0
        if exact_rate < 30:  # Less than 30% exact matches
            issues.append({
                'type': 'low_exact_match_rate',
                'severity': 'medium',
                'rate': exact_rate,
                'description': f'Only {exact_rate:.1f}% of entities have exact matches',
                'recommendation': 'Review data standardization and UID generation logic'
            })
        
        # Log identified issues
        if issues:
            self.logger.warning(f"Identified {len(issues)} data quality issues:")
            for issue in issues:
                self.logger.warning(f"  {issue['type']}: {issue['description']}")
        else:
            self.logger.info("No significant data quality issues identified")
        
        return issues
    
    def _create_entity_key(self, entity: EntityRecord) -> str:
        """
        Create a unique key for an entity.
        
        Args:
            entity: EntityRecord to create key for
            
        Returns:
            Unique string key for the entity
        """
        return f"{entity.district}|{entity.block}|{entity.village}"