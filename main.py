"""
Main entry point for LGD mapping application.

This script provides the command-line interface for running the LGD mapping process.
"""

import argparse
import sys
import time
import psutil
import gc
from pathlib import Path
from tqdm import tqdm

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lgd_mapping.config import MappingConfig
from lgd_mapping.logging_config import setup_logging
from lgd_mapping.mapping_engine import MappingEngine
from lgd_mapping.output.output_generator import OutputGenerator
from lgd_mapping.exceptions import MappingProcessError, DataQualityError


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LGD Mapping Application - Map entities to LGD codes"
    )
    
    parser.add_argument(
        "--entities", 
        required=True,
        help="Path to entities CSV file"
    )
    
    parser.add_argument(
        "--codes", 
        required=True,
        help="Path to LGD codes CSV file"
    )
    
    parser.add_argument(
        "--output", 
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[95, 90],
        help="Fuzzy matching thresholds (default: 95 90)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Process data in chunks of this size (for large datasets)"
    )
    
    parser.add_argument(
        "--district-mapping",
        help="Path to JSON file with district name to code mapping"
    )
    
    parser.add_argument(
        "--cleanup-days",
        type=int,
        default=30,
        help="Clean up output files older than this many days (default: 30)"
    )
    
    # Hierarchical mapping configuration
    parser.add_argument(
        "--enable-hierarchical-mapping",
        action="store_true",
        default=True,
        help="Enable full hierarchical mapping (default: enabled)"
    )
    
    parser.add_argument(
        "--disable-hierarchical-mapping",
        action="store_true",
        help="Disable hierarchical mapping (use legacy 3-level mapping)"
    )
    
    parser.add_argument(
        "--enable-state-code-mapping",
        action="store_true",
        help="Enable automatic state code mapping from state names"
    )
    
    parser.add_argument(
        "--disable-district-code-mapping",
        action="store_true",
        help="Disable automatic district code mapping"
    )
    
    parser.add_argument(
        "--disable-block-code-mapping",
        action="store_true",
        help="Disable automatic block code mapping"
    )
    
    parser.add_argument(
        "--disable-gp-code-mapping",
        action="store_true",
        help="Disable automatic GP (Gram Panchayat) code mapping"
    )
    
    # Level-specific fuzzy thresholds
    parser.add_argument(
        "--state-fuzzy-threshold",
        type=int,
        default=85,
        help="Fuzzy matching threshold for state level (0-100, default: 85)"
    )
    
    parser.add_argument(
        "--district-fuzzy-threshold",
        type=int,
        default=85,
        help="Fuzzy matching threshold for district level (0-100, default: 85)"
    )
    
    parser.add_argument(
        "--block-fuzzy-threshold",
        type=int,
        default=90,
        help="Fuzzy matching threshold for block level (0-100, default: 90)"
    )
    
    parser.add_argument(
        "--gp-fuzzy-threshold",
        type=int,
        default=90,
        help="Fuzzy matching threshold for GP level (0-100, default: 90)"
    )
    
    parser.add_argument(
        "--village-fuzzy-threshold",
        type=int,
        default=95,
        help="Fuzzy matching threshold for village level (0-100, default: 95)"
    )
    
    # Hierarchy validation options
    parser.add_argument(
        "--disable-hierarchy-consistency",
        action="store_true",
        help="Disable hierarchical consistency validation"
    )
    
    parser.add_argument(
        "--disallow-partial-hierarchy",
        action="store_true",
        help="Require all hierarchical levels to be present (strict mode)"
    )
    
    return parser.parse_args()


def load_district_mapping(mapping_file: str) -> dict:
    """Load district mapping from JSON file."""
    if not mapping_file or not Path(mapping_file).exists():
        return {}
    
    try:
        import json
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load district mapping from {mapping_file}: {e}")
        return {}


class PerformanceMonitor:
    """Monitor and log performance metrics during application execution."""
    
    def __init__(self, logger=None):
        """Initialize performance monitor."""
        self.logger = logger
        self.process = psutil.Process()
        self.start_time = time.time()
        self.checkpoints = {}
        self.memory_snapshots = []
        
    def log_memory_usage(self, checkpoint_name: str):
        """Log current memory usage."""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        snapshot = {
            'checkpoint': checkpoint_name,
            'timestamp': time.time(),
            'memory_mb': memory_mb,
            'memory_percent': self.process.memory_percent()
        }
        
        self.memory_snapshots.append(snapshot)
        
        if self.logger:
            self.logger.info(f"Memory usage at {checkpoint_name}: {memory_mb:.1f} MB ({snapshot['memory_percent']:.1f}%)")
        else:
            print(f"Memory usage at {checkpoint_name}: {memory_mb:.1f} MB ({snapshot['memory_percent']:.1f}%)")
    
    def start_checkpoint(self, name: str):
        """Start timing a checkpoint."""
        self.checkpoints[name] = {'start': time.time()}
        self.log_memory_usage(f"{name}_start")
    
    def end_checkpoint(self, name: str):
        """End timing a checkpoint."""
        if name in self.checkpoints:
            end_time = time.time()
            duration = end_time - self.checkpoints[name]['start']
            self.checkpoints[name]['end'] = end_time
            self.checkpoints[name]['duration'] = duration
            
            self.log_memory_usage(f"{name}_end")
            
            if self.logger:
                self.logger.info(f"Checkpoint {name} completed in {duration:.2f} seconds")
            else:
                print(f"Checkpoint {name} completed in {duration:.2f} seconds")
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if not self.memory_snapshots:
            return 0.0
        return max(snapshot['memory_mb'] for snapshot in self.memory_snapshots)
    
    def get_memory_growth(self) -> float:
        """Get memory growth from start to end in MB."""
        if len(self.memory_snapshots) < 2:
            return 0.0
        return self.memory_snapshots[-1]['memory_mb'] - self.memory_snapshots[0]['memory_mb']
    
    def force_garbage_collection(self):
        """Force garbage collection and log memory impact."""
        before_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Force garbage collection
        collected = gc.collect()
        
        after_memory = self.process.memory_info().rss / 1024 / 1024
        memory_freed = before_memory - after_memory
        
        if self.logger:
            self.logger.info(f"Garbage collection: freed {memory_freed:.1f} MB, collected {collected} objects")
        else:
            print(f"Garbage collection: freed {memory_freed:.1f} MB, collected {collected} objects")
    
    def get_performance_summary(self) -> dict:
        """Get comprehensive performance summary."""
        total_time = time.time() - self.start_time
        peak_memory = self.get_peak_memory()
        memory_growth = self.get_memory_growth()
        
        return {
            'total_execution_time': total_time,
            'peak_memory_mb': peak_memory,
            'memory_growth_mb': memory_growth,
            'checkpoints': self.checkpoints.copy(),
            'memory_snapshots': len(self.memory_snapshots)
        }


def optimize_for_large_datasets(config, entities_count: int, lgd_count: int) -> dict:
    """Determine optimizations based on dataset size."""
    optimizations = {
        'use_chunking': False,
        'chunk_size': None,
        'enable_gc_optimization': False,
        'use_progress_bars': True,
        'memory_monitoring_interval': 1000
    }
    
    total_records = entities_count + lgd_count
    
    # Enable chunking for large datasets
    if total_records > 100000:
        optimizations['use_chunking'] = True
        optimizations['chunk_size'] = min(10000, entities_count // 10)
        optimizations['enable_gc_optimization'] = True
        optimizations['memory_monitoring_interval'] = 500
        print(f"Large dataset detected ({total_records:,} records). Enabling optimizations:")
        print(f"  - Chunk processing: {optimizations['chunk_size']:,} records per chunk")
        print(f"  - Garbage collection optimization: enabled")
    elif total_records > 50000:
        optimizations['use_chunking'] = True
        optimizations['chunk_size'] = min(5000, entities_count // 5)
        optimizations['memory_monitoring_interval'] = 1000
        print(f"Medium dataset detected ({total_records:,} records). Enabling chunk processing.")
    
    return optimizations


def print_processing_summary(results, processing_stats, quality_metrics):
    """Print a summary of processing results to console."""
    print("\n" + "="*60)
    print("LGD MAPPING PROCESS COMPLETED")
    print("="*60)
    
    print(f"\nProcessing Summary:")
    print(f"  Total entities processed: {processing_stats.total_entities:,}")
    print(f"  Processing time: {processing_stats.processing_time:.2f} seconds")
    print(f"  Overall match rate: {processing_stats.get_match_rate():.2f}%")
    
    print(f"\nMatching Results:")
    print(f"  Exact matches: {processing_stats.exact_matches:,} "
          f"({(processing_stats.exact_matches/processing_stats.total_entities*100):.2f}%)")
    
    for threshold, count in processing_stats.fuzzy_matches.items():
        rate = (count / processing_stats.total_entities * 100) if processing_stats.total_entities > 0 else 0
        print(f"  Fuzzy matches ({threshold}%): {count:,} ({rate:.2f}%)")
    
    print(f"  Unmatched: {processing_stats.unmatched:,} "
          f"({(processing_stats.unmatched/processing_stats.total_entities*100):.2f}%)")
    
    if quality_metrics:
        unmatched_with_alt = quality_metrics['unmatched_analysis']['with_alternatives']
        unmatched_without_alt = quality_metrics['unmatched_analysis']['without_alternatives']
        
        print(f"\nUnmatched Analysis:")
        print(f"  With alternatives: {unmatched_with_alt:,}")
        print(f"  Without alternatives: {unmatched_without_alt:,}")
        
        if 'data_quality_indicators' in quality_metrics:
            district_coverage = quality_metrics['data_quality_indicators']['district_code_coverage']
            print(f"  District code coverage: {district_coverage:.1f}%")


def main():
    """Main application entry point."""
    start_time = time.time()
    args = parse_arguments()
    
    # Initialize performance monitor
    perf_monitor = None
    
    try:
        print("LGD Mapping Application Starting...")
        print(f"Entities file: {args.entities}")
        print(f"Codes file: {args.codes}")
        print(f"Output directory: {args.output}")
        
        # Load district mapping if provided
        district_mapping = load_district_mapping(args.district_mapping) if args.district_mapping else {}
        
        # Determine hierarchical mapping settings from arguments
        enable_hierarchical = args.enable_hierarchical_mapping and not args.disable_hierarchical_mapping
        enable_district_mapping = not args.disable_district_code_mapping
        enable_block_mapping = not args.disable_block_code_mapping
        enable_gp_mapping = not args.disable_gp_code_mapping
        enforce_consistency = not args.disable_hierarchy_consistency
        allow_partial = not args.disallow_partial_hierarchy
        
        # Create configuration
        config = MappingConfig(
            input_entities_file=args.entities,
            input_codes_file=args.codes,
            output_directory=args.output,
            fuzzy_thresholds=args.thresholds,
            district_code_mapping=district_mapping,
            log_level=args.log_level,
            chunk_size=args.chunk_size,
            # Hierarchical mapping configuration
            enable_hierarchical_mapping=enable_hierarchical,
            enable_state_code_mapping=args.enable_state_code_mapping,
            enable_district_code_mapping=enable_district_mapping,
            enable_block_code_mapping=enable_block_mapping,
            enable_gp_code_mapping=enable_gp_mapping,
            # Level-specific fuzzy thresholds
            state_fuzzy_threshold=args.state_fuzzy_threshold,
            district_fuzzy_threshold=args.district_fuzzy_threshold,
            block_fuzzy_threshold=args.block_fuzzy_threshold,
            gp_fuzzy_threshold=args.gp_fuzzy_threshold,
            village_fuzzy_threshold=args.village_fuzzy_threshold,
            # Hierarchy validation
            enforce_hierarchy_consistency=enforce_consistency,
            allow_partial_hierarchy=allow_partial
        )
        
        # Set up logging
        logger = setup_logging(config)
        
        # Initialize performance monitoring
        perf_monitor = PerformanceMonitor(logger.logger)
        perf_monitor.start_checkpoint("initialization")
        
        logger.info("LGD Mapping Application initialized")
        logger.info(f"Configuration: {config.to_dict()}")
        
        # Initialize mapping engine
        print("Initializing mapping engine...")
        mapping_engine = MappingEngine(config, logger)
        
        # Initialize output generator
        output_generator = OutputGenerator(config, logger)
        
        # Validate output directory
        if not output_generator.validate_output_directory():
            raise MappingProcessError("Output directory is not writable", strategy="initialization")
        
        perf_monitor.end_checkpoint("initialization")
        
        # Get dataset sizes for optimization
        perf_monitor.start_checkpoint("data_size_analysis")
        
        # Quick file size check for optimization decisions
        import pandas as pd
        try:
            entities_sample = pd.read_csv(config.input_entities_file, nrows=1)
            entities_count = sum(1 for _ in open(config.input_entities_file)) - 1  # Subtract header
            
            lgd_sample = pd.read_csv(config.input_codes_file, nrows=1)
            lgd_count = sum(1 for _ in open(config.input_codes_file)) - 1  # Subtract header
            
            print(f"Dataset sizes: {entities_count:,} entities, {lgd_count:,} LGD codes")
            
            # Determine optimizations
            optimizations = optimize_for_large_datasets(config, entities_count, lgd_count)
            
            # Apply chunk size optimization if not already set
            if optimizations['use_chunking'] and not config.chunk_size:
                config.chunk_size = optimizations['chunk_size']
                logger.info(f"Applied automatic chunk size: {config.chunk_size:,}")
                
        except Exception as e:
            logger.warning(f"Could not analyze dataset sizes: {e}")
            entities_count = 0
            lgd_count = 0
            optimizations = {'enable_gc_optimization': False}
        
        perf_monitor.end_checkpoint("data_size_analysis")
        
        # Run complete mapping process with performance monitoring
        perf_monitor.start_checkpoint("mapping_process")
        print("Starting mapping process...")
        
        # Enable garbage collection optimization for large datasets
        if optimizations.get('enable_gc_optimization', False):
            perf_monitor.force_garbage_collection()
        
        results, processing_stats = mapping_engine.run_complete_mapping()
        
        perf_monitor.end_checkpoint("mapping_process")
        
        # Generate quality metrics
        perf_monitor.start_checkpoint("quality_analysis")
        print("Calculating quality metrics...")
        from lgd_mapping.mapping_engine import ResultsAggregator
        aggregator = ResultsAggregator(logger)
        quality_metrics = aggregator.generate_quality_report(results)
        perf_monitor.end_checkpoint("quality_analysis")
        
        # Generate output files
        perf_monitor.start_checkpoint("output_generation")
        print("Generating output files...")
        generated_files = output_generator.generate_all_outputs(results, processing_stats)
        perf_monitor.end_checkpoint("output_generation")
        
        # Clean up old files if requested
        if args.cleanup_days > 0:
            perf_monitor.start_checkpoint("file_cleanup")
            print(f"Cleaning up files older than {args.cleanup_days} days...")
            cleaned_count = output_generator.cleanup_old_files(args.cleanup_days)
            if cleaned_count > 0:
                print(f"Cleaned up {cleaned_count} old files")
            perf_monitor.end_checkpoint("file_cleanup")
        
        # Final garbage collection for large datasets
        if optimizations.get('enable_gc_optimization', False):
            perf_monitor.force_garbage_collection()
        
        # Print processing summary
        print_processing_summary(results, processing_stats, quality_metrics)
        
        # Print generated files
        print(f"\nGenerated Output Files:")
        for file_type, file_path in generated_files.items():
            if file_path:
                print(f"  {file_type}: {Path(file_path).name}")
        
        # Log data quality information
        if mapping_engine.has_critical_quality_issues():
            print(f"\nWarning: Critical data quality issues detected!")
            print(f"Data quality score: {mapping_engine.get_data_quality_score():.1f}/100")
            print("Please review the processing log for details.")
        
        # Print performance summary
        if perf_monitor:
            perf_summary = perf_monitor.get_performance_summary()
            print(f"\nPerformance Summary:")
            print(f"  Total execution time: {perf_summary['total_execution_time']:.2f} seconds")
            print(f"  Peak memory usage: {perf_summary['peak_memory_mb']:.1f} MB")
            print(f"  Memory growth: {perf_summary['memory_growth_mb']:.1f} MB")
            
            # Log detailed checkpoint timings
            if perf_summary['checkpoints']:
                print(f"  Phase timings:")
                for checkpoint, timing in perf_summary['checkpoints'].items():
                    if 'duration' in timing:
                        print(f"    {checkpoint}: {timing['duration']:.2f}s")
            
            # Calculate processing rate
            if entities_count > 0 and 'mapping_process' in perf_summary['checkpoints']:
                mapping_duration = perf_summary['checkpoints']['mapping_process'].get('duration', 0)
                if mapping_duration > 0:
                    rate = entities_count / mapping_duration
                    print(f"  Processing rate: {rate:.0f} entities/second")
        
        print("LGD Mapping Process completed successfully!")
        
        logger.info("Application completed successfully")
        
    except DataQualityError as e:
        error_msg = f"Data Quality Error: {e}"
        print(f"\nError: {error_msg}", file=sys.stderr)
        if hasattr(e, 'recommendations') and e.recommendations:
            print("Recommendations:", file=sys.stderr)
            for rec in e.recommendations:
                print(f"  - {rec}", file=sys.stderr)
        sys.exit(2)
        
    except MappingProcessError as e:
        error_msg = f"Mapping Process Error: {e}"
        print(f"\nError: {error_msg}", file=sys.stderr)
        if hasattr(e, 'strategy'):
            print(f"Failed during: {e.strategy}", file=sys.stderr)
        sys.exit(3)
        
    except FileNotFoundError as e:
        print(f"\nFile Error: {e}", file=sys.stderr)
        print("Please check that all input files exist and are accessible.", file=sys.stderr)
        sys.exit(4)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user", file=sys.stderr)
        sys.exit(130)
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"\nError: {error_msg}", file=sys.stderr)
        print("Please check the log files for more details.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()