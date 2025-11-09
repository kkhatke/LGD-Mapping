"""
Output generation and file management for LGD mapping application.

This module provides the OutputGenerator class for creating organized output files
with descriptive naming, timestamps, and proper CSV formatting for mapping results.
"""

import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import csv

from ..models import MappingResult, EntityRecord, LGDRecord
from ..config import MappingConfig, ProcessingStats
from ..logging_config import MappingLogger


class OutputGenerator:
    """
    Generates organized output files for mapping results.
    
    This class creates separate output files for each matching strategy result
    with descriptive file naming, timestamps, and proper CSV formatting.
    """
    
    def __init__(self, config: MappingConfig, logger: Optional[MappingLogger] = None):
        """
        Initialize the OutputGenerator.
        
        Args:
            config: Configuration object with output directory and settings
            logger: Optional logger instance for logging operations
        """
        self.config = config
        self.logger = logger or MappingLogger()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output directory exists
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        # Define output file patterns
        self.file_patterns = {
            'exact': 'exact_matches_{timestamp}.csv',
            'fuzzy_95': 'fuzzy_95_matches_{timestamp}.csv',
            'fuzzy_90': 'fuzzy_90_matches_{timestamp}.csv',
            'unmatched_with_alternatives': 'unmatched_with_alternatives_{timestamp}.csv',
            'unmatched_no_alternatives': 'unmatched_no_alternatives_{timestamp}.csv',
            'all_results': 'all_mapping_results_{timestamp}.csv',
            'summary_report': 'mapping_summary_report_{timestamp}.txt'
        }
    
    def generate_all_outputs(self, results: List[MappingResult], 
                           processing_stats: ProcessingStats) -> Dict[str, str]:
        """
        Generate all output files for mapping results.
        
        Args:
            results: List of all mapping results
            processing_stats: Processing statistics object
            
        Returns:
            Dictionary mapping output type to generated file path
        """
        self.logger.info("Starting output file generation")
        generated_files = {}
        
        try:
            # Generate separate files for each match type
            generated_files.update(self._generate_match_type_files(results))
            
            # Generate comprehensive results file
            all_results_file = self._generate_all_results_file(results)
            generated_files['all_results'] = all_results_file
            
            # Generate summary report
            summary_file = self._generate_summary_report(results, processing_stats)
            generated_files['summary_report'] = summary_file
            
            self.logger.info(f"Generated {len(generated_files)} output files")
            return generated_files
            
        except Exception as e:
            self.logger.error(f"Error generating output files: {e}")
            raise
    
    def _generate_match_type_files(self, results: List[MappingResult]) -> Dict[str, str]:
        """
        Generate separate CSV files for each match type.
        
        Args:
            results: List of all mapping results
            
        Returns:
            Dictionary mapping match type to generated file path
        """
        generated_files = {}
        
        # Group results by match type
        results_by_type = self._group_results_by_type(results)
        
        # Generate file for each match type
        for match_type, type_results in results_by_type.items():
            if not type_results:
                self.logger.info(f"No results for match type: {match_type}")
                continue
            
            file_path = self._generate_match_type_file(match_type, type_results)
            generated_files[match_type] = file_path
            self.logger.info(f"Generated {match_type} file: {file_path} ({len(type_results)} records)")
        
        return generated_files
    
    def _generate_match_type_file(self, match_type: str, results: List[MappingResult]) -> str:
        """
        Generate CSV file for a specific match type.
        
        Args:
            match_type: Type of match ('exact', 'fuzzy_95', etc.)
            results: List of results for this match type
            
        Returns:
            Path to generated file
        """
        # Determine file pattern based on match type
        if match_type == 'unmatched':
            # Split unmatched into with/without alternatives
            with_alternatives = [r for r in results if r.alternative_matches]
            without_alternatives = [r for r in results if not r.alternative_matches]
            
            files_generated = []
            
            if with_alternatives:
                file_path = self._create_unmatched_file('unmatched_with_alternatives', 
                                                      with_alternatives, include_alternatives=True)
                files_generated.append(file_path)
            
            if without_alternatives:
                file_path = self._create_unmatched_file('unmatched_no_alternatives', 
                                                      without_alternatives, include_alternatives=False)
                files_generated.append(file_path)
            
            return files_generated[0] if files_generated else ""
        else:
            # Regular match type file
            file_pattern = self.file_patterns.get(match_type, f'{match_type}_matches_{{timestamp}}.csv')
            filename = file_pattern.format(timestamp=self.timestamp)
            file_path = os.path.join(self.config.output_directory, filename)
            
            # Create DataFrame and save
            df = self._create_results_dataframe(results, include_alternatives=False)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            return file_path
    
    def _create_unmatched_file(self, file_type: str, results: List[MappingResult], 
                              include_alternatives: bool) -> str:
        """
        Create CSV file for unmatched results.
        
        Args:
            file_type: Type of unmatched file ('unmatched_with_alternatives' or 'unmatched_no_alternatives')
            results: List of unmatched results
            include_alternatives: Whether to include alternative matches column
            
        Returns:
            Path to generated file
        """
        filename = self.file_patterns[file_type].format(timestamp=self.timestamp)
        file_path = os.path.join(self.config.output_directory, filename)
        
        df = self._create_results_dataframe(results, include_alternatives=include_alternatives)
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        return file_path
    
    def _generate_all_results_file(self, results: List[MappingResult]) -> str:
        """
        Generate comprehensive CSV file with all mapping results.
        
        Args:
            results: List of all mapping results
            
        Returns:
            Path to generated file
        """
        filename = self.file_patterns['all_results'].format(timestamp=self.timestamp)
        file_path = os.path.join(self.config.output_directory, filename)
        
        # Create comprehensive DataFrame with all columns
        df = self._create_results_dataframe(results, include_alternatives=True, comprehensive=True)
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        self.logger.info(f"Generated comprehensive results file: {file_path} ({len(results)} records)")
        return file_path
    
    def _create_results_dataframe(self, results: List[MappingResult], 
                                 include_alternatives: bool = False,
                                 comprehensive: bool = False) -> pd.DataFrame:
        """
        Create pandas DataFrame from mapping results.
        
        Args:
            results: List of mapping results
            include_alternatives: Whether to include alternative matches
            comprehensive: Whether to include all possible columns
            
        Returns:
            DataFrame with formatted results
        """
        # Define base columns with hierarchical support
        base_columns = [
            'source_state', 'source_state_code',
            'source_district', 'source_district_code',
            'source_subdistrict', 'source_subdistrict_code',
            'source_block', 'source_block_code',
            'source_gp', 'source_gp_code',
            'source_village', 'source_village_code',
            'match_type', 'match_score'
        ]
        
        # Add LGD columns for matched results with full hierarchy
        lgd_columns = [
            'lgd_state_code', 'lgd_state',
            'lgd_district_code', 'lgd_district',
            'lgd_subdistrict_code', 'lgd_subdistrict',
            'lgd_block_code', 'lgd_block',
            'lgd_gp_code', 'lgd_gp',
            'lgd_village_code', 'lgd_village'
        ]
        
        # Add hierarchical metadata columns
        hierarchy_columns = [
            'hierarchy_depth', 'hierarchy_levels',
            'hierarchy_match_summary', 'hierarchy_confidence'
        ]
        
        # Add alternative matches column if requested
        alt_columns = ['alternative_matches'] if include_alternatives else []
        
        # Add comprehensive columns if requested
        comp_columns = ['confidence_level', 'is_matched'] if comprehensive else []
        
        all_columns = base_columns + lgd_columns + hierarchy_columns + alt_columns + comp_columns
        
        # Handle empty results - create empty DataFrame with proper columns
        if not results:
            return pd.DataFrame(columns=all_columns)
        
        # Create data rows
        data_rows = []
        for result in results:
            row = self._create_result_row(result, all_columns)
            data_rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=all_columns)
        
        # Apply data type conversions
        df = self._apply_dataframe_formatting(df)
        
        return df
    
    def _create_result_row(self, result: MappingResult, columns: List[str]) -> List[Any]:
        """
        Create a data row for a single mapping result.
        
        Args:
            result: MappingResult object
            columns: List of column names to include
            
        Returns:
            List of values for the row
        """
        row = []
        
        for column in columns:
            # Source entity hierarchical fields
            if column == 'source_state':
                row.append(result.entity.state)
            elif column == 'source_state_code':
                row.append(result.entity.state_code)
            elif column == 'source_district':
                row.append(result.entity.district)
            elif column == 'source_district_code':
                row.append(result.entity.district_code)
            elif column == 'source_subdistrict':
                row.append(result.entity.subdistrict)
            elif column == 'source_subdistrict_code':
                row.append(result.entity.subdistrict_code)
            elif column == 'source_block':
                row.append(result.entity.block)
            elif column == 'source_block_code':
                row.append(result.entity.block_code)
            elif column == 'source_gp':
                row.append(result.entity.gp)
            elif column == 'source_gp_code':
                row.append(result.entity.gp_code)
            elif column == 'source_village':
                row.append(result.entity.village)
            elif column == 'source_village_code':
                row.append(result.entity.village_code)
            
            # Match information
            elif column == 'match_type':
                row.append(result.match_type)
            elif column == 'match_score':
                row.append(result.match_score)
            
            # LGD match hierarchical fields
            elif column == 'lgd_state_code':
                row.append(result.lgd_match.state_code if result.lgd_match else None)
            elif column == 'lgd_state':
                row.append(result.lgd_match.state if result.lgd_match else None)
            elif column == 'lgd_district_code':
                row.append(result.lgd_match.district_code if result.lgd_match else None)
            elif column == 'lgd_district':
                row.append(result.lgd_match.district if result.lgd_match else None)
            elif column == 'lgd_subdistrict_code':
                row.append(result.lgd_match.subdistrict_code if result.lgd_match else None)
            elif column == 'lgd_subdistrict':
                row.append(result.lgd_match.subdistrict if result.lgd_match else None)
            elif column == 'lgd_block_code':
                row.append(result.lgd_match.block_code if result.lgd_match else None)
            elif column == 'lgd_block':
                row.append(result.lgd_match.block if result.lgd_match else None)
            elif column == 'lgd_gp_code':
                row.append(result.lgd_match.gp_code if result.lgd_match else None)
            elif column == 'lgd_gp':
                row.append(result.lgd_match.gp if result.lgd_match else None)
            elif column == 'lgd_village_code':
                row.append(result.lgd_match.village_code if result.lgd_match else None)
            elif column == 'lgd_village':
                row.append(result.lgd_match.village if result.lgd_match else None)
            
            # Hierarchical metadata
            elif column == 'hierarchy_depth':
                row.append(result.entity.hierarchy_depth)
            elif column == 'hierarchy_levels':
                row.append(', '.join(result.entity.hierarchy_levels))
            elif column == 'hierarchy_match_summary':
                row.append(result.get_hierarchy_match_summary())
            elif column == 'hierarchy_confidence':
                row.append(result.hierarchy_confidence)
            
            # Other columns
            elif column == 'alternative_matches':
                row.append('; '.join(result.alternative_matches) if result.alternative_matches else '')
            elif column == 'confidence_level':
                row.append(result.get_confidence_level())
            elif column == 'is_matched':
                row.append(result.is_matched())
            else:
                row.append(None)
        
        return row
    
    def _apply_dataframe_formatting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply proper data type formatting to DataFrame.
        
        Args:
            df: DataFrame to format
            
        Returns:
            Formatted DataFrame
        """
        # Convert numeric columns to proper types (including hierarchical codes)
        numeric_columns = [
            'source_state_code', 'source_district_code', 'source_subdistrict_code',
            'source_block_code', 'source_gp_code', 'source_village_code',
            'lgd_state_code', 'lgd_district_code', 'lgd_subdistrict_code',
            'lgd_block_code', 'lgd_gp_code', 'lgd_village_code',
            'match_score', 'hierarchy_depth', 'hierarchy_confidence'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        boolean_columns = ['is_matched']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Ensure string columns are properly formatted (including hierarchical fields)
        string_columns = [
            'source_state', 'source_district', 'source_subdistrict', 'source_block',
            'source_gp', 'source_village',
            'lgd_state', 'lgd_district', 'lgd_subdistrict', 'lgd_block',
            'lgd_gp', 'lgd_village',
            'match_type', 'confidence_level', 'alternative_matches',
            'hierarchy_levels', 'hierarchy_match_summary'
        ]
        
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '')
                df[col] = df[col].replace('None', '')
        
        return df
    
    def _group_results_by_type(self, results: List[MappingResult]) -> Dict[str, List[MappingResult]]:
        """
        Group results by match type.
        
        Args:
            results: List of all mapping results
            
        Returns:
            Dictionary mapping match type to list of results
        """
        grouped = {}
        
        for result in results:
            match_type = result.match_type
            if match_type not in grouped:
                grouped[match_type] = []
            grouped[match_type].append(result)
        
        return grouped
    
    def _generate_summary_report(self, results: List[MappingResult], 
                               processing_stats: ProcessingStats) -> str:
        """
        Generate comprehensive text summary report.
        
        Args:
            results: List of all mapping results
            processing_stats: Processing statistics
            
        Returns:
            Path to generated summary report file
        """
        filename = self.file_patterns['summary_report'].format(timestamp=self.timestamp)
        file_path = os.path.join(self.config.output_directory, filename)
        
        # Import here to avoid circular imports
        from ..mapping_engine import ResultsAggregator
        
        # Generate quality metrics
        aggregator = ResultsAggregator(self.logger)
        quality_metrics = aggregator.generate_quality_report(results)
        
        # Create comprehensive report
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("LGD MAPPING SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic information
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing Time: {processing_stats.processing_time:.2f} seconds\n")
            f.write(f"Configuration:\n")
            f.write(f"  Input Entities: {self.config.input_entities_file}\n")
            f.write(f"  Input Codes: {self.config.input_codes_file}\n")
            f.write(f"  Fuzzy Thresholds: {self.config.fuzzy_thresholds}\n")
            f.write(f"  Chunk Size: {self.config.chunk_size or 'Not specified'}\n\n")
            
            # Processing summary
            f.write("PROCESSING SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Entities Processed: {processing_stats.total_entities:,}\n")
            f.write(f"Overall Match Rate: {processing_stats.get_match_rate():.2f}%\n\n")
            
            # Detailed matching results
            f.write("MATCHING RESULTS\n")
            f.write("-" * 16 + "\n")
            
            # Exact matches
            exact_rate = (processing_stats.exact_matches / processing_stats.total_entities * 100) if processing_stats.total_entities > 0 else 0
            f.write(f"Exact Matches: {processing_stats.exact_matches:,} ({exact_rate:.2f}%)\n")
            
            # Fuzzy matches by threshold
            for threshold, count in processing_stats.fuzzy_matches.items():
                fuzzy_rate = (count / processing_stats.total_entities * 100) if processing_stats.total_entities > 0 else 0
                f.write(f"Fuzzy Matches ({threshold}%): {count:,} ({fuzzy_rate:.2f}%)\n")
            
            # Unmatched
            unmatched_rate = (processing_stats.unmatched / processing_stats.total_entities * 100) if processing_stats.total_entities > 0 else 0
            f.write(f"Unmatched: {processing_stats.unmatched:,} ({unmatched_rate:.2f}%)\n\n")
            
            # Quality metrics
            if quality_metrics:
                f.write("QUALITY ANALYSIS\n")
                f.write("-" * 16 + "\n")
                
                unmatched_analysis = quality_metrics.get('unmatched_analysis', {})
                f.write(f"Unmatched with Alternatives: {unmatched_analysis.get('with_alternatives', 0):,}\n")
                f.write(f"Unmatched without Alternatives: {unmatched_analysis.get('without_alternatives', 0):,}\n")
                f.write(f"Alternative Coverage Rate: {unmatched_analysis.get('alternative_coverage_rate', 0):.2f}%\n\n")
                
                # Confidence distribution
                confidence_dist = quality_metrics.get('confidence_distribution', {})
                f.write("CONFIDENCE DISTRIBUTION\n")
                f.write("-" * 22 + "\n")
                for confidence, count in confidence_dist.items():
                    conf_rate = (count / processing_stats.total_entities * 100) if processing_stats.total_entities > 0 else 0
                    f.write(f"{confidence} Confidence: {count:,} ({conf_rate:.2f}%)\n")
                f.write("\n")
                
                # Data quality indicators
                data_quality = quality_metrics.get('data_quality_indicators', {})
                if data_quality:
                    f.write("DATA QUALITY INDICATORS\n")
                    f.write("-" * 23 + "\n")
                    district_coverage = data_quality.get('district_code_coverage', 0)
                    f.write(f"District Code Coverage: {district_coverage:.2f}%\n\n")
            
            # Hierarchical statistics
            hierarchy_stats = self._calculate_hierarchy_statistics(results)
            if hierarchy_stats:
                f.write("HIERARCHICAL MAPPING STATISTICS\n")
                f.write("-" * 32 + "\n")
                
                # Hierarchy depth distribution
                depth_dist = hierarchy_stats.get('depth_distribution', {})
                if depth_dist:
                    f.write("Hierarchy Depth Distribution:\n")
                    for depth, count in sorted(depth_dist.items()):
                        depth_rate = (count / processing_stats.total_entities * 100) if processing_stats.total_entities > 0 else 0
                        f.write(f"  {depth}-level hierarchy: {count:,} entities ({depth_rate:.2f}%)\n")
                    f.write("\n")
                
                # Level presence statistics
                level_presence = hierarchy_stats.get('level_presence', {})
                if level_presence:
                    f.write("Hierarchical Level Presence:\n")
                    for level, count in level_presence.items():
                        presence_rate = (count / processing_stats.total_entities * 100) if processing_stats.total_entities > 0 else 0
                        f.write(f"  {level.capitalize()}: {count:,} entities ({presence_rate:.2f}%)\n")
                    f.write("\n")
                
                # Hierarchical match quality
                match_quality = hierarchy_stats.get('hierarchical_match_quality', {})
                if match_quality:
                    f.write("Hierarchical Match Quality:\n")
                    avg_confidence = match_quality.get('average_hierarchy_confidence', 0)
                    f.write(f"  Average Hierarchy Confidence: {avg_confidence:.3f}\n")
                    
                    level_match_rates = match_quality.get('level_match_rates', {})
                    if level_match_rates:
                        f.write("  Match Rates by Level:\n")
                        for level, rate in level_match_rates.items():
                            f.write(f"    {level.capitalize()}: {rate:.2f}%\n")
                    f.write("\n")
                
                # Code coverage by level
                code_coverage = hierarchy_stats.get('code_coverage', {})
                if code_coverage:
                    f.write("Code Coverage by Level:\n")
                    for level, coverage in code_coverage.items():
                        f.write(f"  {level.capitalize()}: {coverage:.2f}%\n")
                    f.write("\n")
            
            # File output summary
            f.write("OUTPUT FILES GENERATED\n")
            f.write("-" * 22 + "\n")
            
            # Group results by type for file summary
            results_by_type = self._group_results_by_type(results)
            for match_type, type_results in results_by_type.items():
                count = len(type_results)
                if count > 0:
                    if match_type == 'unmatched':
                        with_alt = len([r for r in type_results if r.alternative_matches])
                        without_alt = count - with_alt
                        if with_alt > 0:
                            f.write(f"unmatched_with_alternatives.csv: {with_alt:,} records\n")
                        if without_alt > 0:
                            f.write(f"unmatched_no_alternatives.csv: {without_alt:,} records\n")
                    else:
                        f.write(f"{match_type}_matches.csv: {count:,} records\n")
            
            f.write(f"all_mapping_results.csv: {len(results):,} records\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            if processing_stats.unmatched > 0:
                unmatched_rate = (processing_stats.unmatched / processing_stats.total_entities * 100)
                if unmatched_rate > 20:
                    f.write("• High unmatched rate detected. Consider:\n")
                    f.write("  - Reviewing entity data quality\n")
                    f.write("  - Updating LGD reference data\n")
                    f.write("  - Adjusting fuzzy matching thresholds\n")
                elif unmatched_rate > 10:
                    f.write("• Moderate unmatched rate. Review unmatched records with alternatives.\n")
                else:
                    f.write("• Good match rate achieved.\n")
            
            if quality_metrics:
                unmatched_no_alt = quality_metrics.get('unmatched_analysis', {}).get('without_alternatives', 0)
                if unmatched_no_alt > processing_stats.total_entities * 0.05:  # More than 5%
                    f.write("• Significant number of entities have no match alternatives.\n")
                    f.write("  Consider expanding LGD reference data coverage.\n")
            
            # Performance notes
            entities_per_second = processing_stats.total_entities / processing_stats.processing_time if processing_stats.processing_time > 0 else 0
            f.write(f"\nPERFORMANCE\n")
            f.write("-" * 11 + "\n")
            f.write(f"Processing Rate: {entities_per_second:.0f} entities/second\n")
            
            if processing_stats.processing_time > 300:  # More than 5 minutes
                f.write("• Consider using chunk processing for better memory management.\n")
            
            f.write(f"\nReport completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.logger.info(f"Generated summary report: {file_path}")
        return file_path
    
    def get_output_file_info(self) -> Dict[str, str]:
        """
        Get information about output file naming patterns.
        
        Returns:
            Dictionary with file pattern information
        """
        return {
            'timestamp': self.timestamp,
            'output_directory': self.config.output_directory,
            'file_patterns': self.file_patterns.copy()
        }
    
    def validate_output_directory(self) -> bool:
        """
        Validate that output directory is writable.
        
        Returns:
            True if directory is writable, False otherwise
        """
        try:
            # Test write access by creating a temporary file
            test_file = os.path.join(self.config.output_directory, f'test_write_{self.timestamp}.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except Exception as e:
            self.logger.error(f"Output directory not writable: {e}")
            return False
    
    def _calculate_hierarchy_statistics(self, results: List[MappingResult]) -> Dict[str, Any]:
        """
        Calculate hierarchical mapping statistics from results.
        
        Args:
            results: List of all mapping results
            
        Returns:
            Dictionary containing hierarchical statistics
        """
        if not results:
            return {}
        
        stats = {
            'depth_distribution': {},
            'level_presence': {},
            'hierarchical_match_quality': {},
            'code_coverage': {}
        }
        
        # Track hierarchy depth distribution
        for result in results:
            depth = result.entity.hierarchy_depth
            stats['depth_distribution'][depth] = stats['depth_distribution'].get(depth, 0) + 1
        
        # Track level presence
        level_counts = {
            'state': 0,
            'district': 0,
            'subdistrict': 0,
            'block': 0,
            'gp': 0,
            'village': 0
        }
        
        # Track code coverage
        code_counts = {
            'state_code': 0,
            'district_code': 0,
            'subdistrict_code': 0,
            'block_code': 0,
            'gp_code': 0,
            'village_code': 0
        }
        
        # Track hierarchical match details
        level_match_counts = {
            'state': {'total': 0, 'exact': 0, 'fuzzy': 0},
            'district': {'total': 0, 'exact': 0, 'fuzzy': 0},
            'subdistrict': {'total': 0, 'exact': 0, 'fuzzy': 0},
            'block': {'total': 0, 'exact': 0, 'fuzzy': 0},
            'gp': {'total': 0, 'exact': 0, 'fuzzy': 0},
            'village': {'total': 0, 'exact': 0, 'fuzzy': 0}
        }
        
        total_hierarchy_confidence = 0.0
        matched_count = 0
        
        for result in results:
            # Count level presence
            for level in result.entity.hierarchy_levels:
                if level in level_counts:
                    level_counts[level] += 1
            
            # Count code coverage
            if result.entity.state_code is not None:
                code_counts['state_code'] += 1
            if result.entity.district_code is not None:
                code_counts['district_code'] += 1
            if result.entity.subdistrict_code is not None:
                code_counts['subdistrict_code'] += 1
            if result.entity.block_code is not None:
                code_counts['block_code'] += 1
            if result.entity.gp_code is not None:
                code_counts['gp_code'] += 1
            if result.entity.village_code is not None:
                code_counts['village_code'] += 1
            
            # Track hierarchical match quality
            if result.hierarchy_match_details:
                for level, match_type in result.hierarchy_match_details.items():
                    if level in level_match_counts:
                        level_match_counts[level]['total'] += 1
                        if match_type == 'exact':
                            level_match_counts[level]['exact'] += 1
                        elif match_type.startswith('fuzzy_'):
                            level_match_counts[level]['fuzzy'] += 1
            
            # Track hierarchy confidence
            if result.is_matched():
                total_hierarchy_confidence += result.hierarchy_confidence
                matched_count += 1
        
        # Calculate level presence percentages
        total_entities = len(results)
        stats['level_presence'] = level_counts
        
        # Calculate code coverage percentages
        code_coverage = {}
        for code_field, count in code_counts.items():
            level_name = code_field.replace('_code', '')
            if level_counts.get(level_name, 0) > 0:
                coverage = (count / level_counts[level_name]) * 100
                code_coverage[level_name] = coverage
        stats['code_coverage'] = code_coverage
        
        # Calculate hierarchical match quality metrics
        avg_confidence = (total_hierarchy_confidence / matched_count) if matched_count > 0 else 0.0
        
        level_match_rates = {}
        for level, counts in level_match_counts.items():
            if counts['total'] > 0:
                exact_rate = (counts['exact'] / counts['total']) * 100
                level_match_rates[level] = exact_rate
        
        stats['hierarchical_match_quality'] = {
            'average_hierarchy_confidence': avg_confidence,
            'level_match_rates': level_match_rates
        }
        
        return stats
    
    def cleanup_old_files(self, days_to_keep: int = 30) -> int:
        """
        Clean up old output files from the output directory.
        
        Args:
            days_to_keep: Number of days of files to keep
            
        Returns:
            Number of files cleaned up
        """
        if days_to_keep <= 0:
            self.logger.warning("Invalid days_to_keep value, skipping cleanup")
            return 0
        
        try:
            output_path = Path(self.config.output_directory)
            current_time = datetime.now()
            files_cleaned = 0
            
            for file_path in output_path.glob('*'):
                if file_path.is_file():
                    # Check file age
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.days > days_to_keep:
                        file_path.unlink()
                        files_cleaned += 1
                        self.logger.info(f"Cleaned up old file: {file_path.name}")
            
            self.logger.info(f"Cleaned up {files_cleaned} old files")
            return files_cleaned
            
        except Exception as e:
            self.logger.error(f"Error during file cleanup: {e}")
            return 0