#!/usr/bin/env python3
"""
Performance benchmarking script for LGD Mapping application.

This script provides comprehensive performance testing and benchmarking capabilities
for the LGD mapping system, including:
- Processing speed benchmarks
- Memory usage profiling
- Scalability testing with different dataset sizes
- Component-level performance analysis
- Comparison of matching strategies
"""

import sys
import time
import psutil
import gc
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from lgd_mapping.config import MappingConfig
from lgd_mapping.logging_config import setup_logging
from lgd_mapping.mapping_engine import MappingEngine
from lgd_mapping.data_loader import DataLoader
from lgd_mapping.matching.exact_matcher import ExactMatcher
from lgd_mapping.matching.fuzzy_matcher import FuzzyMatcher


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = 0.0
        self.memory_start = 0.0
        self.memory_end = 0.0
        self.memory_peak = 0.0
        self.records_processed = 0
        self.success = False
        self.error = None
        self.metrics = {}
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'duration_seconds': self.duration,
            'memory_start_mb': self.memory_start,
            'memory_end_mb': self.memory_end,
            'memory_peak_mb': self.memory_peak,
            'memory_growth_mb': self.memory_end - self.memory_start,
            'records_processed': self.records_processed,
            'records_per_second': self.records_processed / self.duration if self.duration > 0 else 0,
            'success': self.success,
            'error': str(self.error) if self.error else None,
            'metrics': self.metrics
        }


class PerformanceBenchmark:
    """Main benchmarking class for LGD mapping application."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.process = psutil.Process()
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def run_benchmark(self, name: str, func, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark test."""
        result = BenchmarkResult(name)
        
        print(f"\n{'='*60}")
        print(f"Running benchmark: {name}")
        print(f"{'='*60}")
        
        # Force garbage collection before benchmark
        gc.collect()
        
        result.memory_start = self.get_memory_usage()
        result.start_time = time.time()
        
        try:
            # Run the benchmark function
            benchmark_output = func(*args, **kwargs)
            
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            result.memory_end = self.get_memory_usage()
            result.memory_peak = result.memory_end  # Simplified, could use memory_profiler for accurate peak
            result.success = True
            
            # Extract metrics from output if available
            if isinstance(benchmark_output, dict):
                result.metrics = benchmark_output
                result.records_processed = benchmark_output.get('records_processed', 0)
            elif isinstance(benchmark_output, int):
                result.records_processed = benchmark_output
            
            print(f"✓ Completed in {result.duration:.2f}s")
            print(f"  Memory: {result.memory_start:.1f} MB → {result.memory_end:.1f} MB "
                  f"(+{result.memory_end - result.memory_start:.1f} MB)")
            if result.records_processed > 0:
                print(f"  Processed: {result.records_processed:,} records")
                print(f"  Rate: {result.records_processed / result.duration:.0f} records/second")
            
        except Exception as e:
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            result.memory_end = self.get_memory_usage()
            result.success = False
            result.error = e
            print(f"✗ Failed: {e}")
        
        self.results.append(result)
        return result
    
    def benchmark_data_loading(self, entities_file: str, codes_file: str) -> dict:
        """Benchmark data loading performance."""
        config = MappingConfig(
            input_entities_file=entities_file,
            input_codes_file=codes_file,
            output_directory=str(self.output_dir)
        )
        
        loader = DataLoader(config)
        
        start = time.time()
        entities_df = loader.load_entities()
        entities_time = time.time() - start
        
        start = time.time()
        codes_df = loader.load_lgd_codes()
        codes_time = time.time() - start
        
        return {
            'records_processed': len(entities_df) + len(codes_df),
            'entities_count': len(entities_df),
            'codes_count': len(codes_df),
            'entities_load_time': entities_time,
            'codes_load_time': codes_time,
            'total_load_time': entities_time + codes_time
        }
    
    def benchmark_exact_matching(self, entities_file: str, codes_file: str) -> dict:
        """Benchmark exact matching performance."""
        config = MappingConfig(
            input_entities_file=entities_file,
            input_codes_file=codes_file,
            output_directory=str(self.output_dir)
        )
        
        logger = setup_logging(config)
        loader = DataLoader(config, logger)
        
        entities_df = loader.load_entities()
        codes_df = loader.load_lgd_codes()
        
        matcher = ExactMatcher(logger)
        
        start = time.time()
        matched_df = matcher.match_by_uid(entities_df, codes_df)
        match_time = time.time() - start
        
        match_count = len(matched_df[matched_df['match_type'] == 'exact'])
        
        return {
            'records_processed': len(entities_df),
            'matches_found': match_count,
            'match_rate': (match_count / len(entities_df) * 100) if len(entities_df) > 0 else 0,
            'matching_time': match_time
        }
    
    def benchmark_fuzzy_matching(self, entities_file: str, codes_file: str, threshold: int = 95) -> dict:
        """Benchmark fuzzy matching performance."""
        config = MappingConfig(
            input_entities_file=entities_file,
            input_codes_file=codes_file,
            output_directory=str(self.output_dir),
            fuzzy_thresholds=[threshold]
        )
        
        logger = setup_logging(config)
        loader = DataLoader(config, logger)
        
        entities_df = loader.load_entities()
        codes_df = loader.load_lgd_codes()
        
        # Create a sample of unmatched records for fuzzy matching
        sample_size = min(1000, len(entities_df))
        sample_df = entities_df.head(sample_size).copy()
        
        matcher = FuzzyMatcher(threshold, logger)
        
        start = time.time()
        matched_df = matcher.fuzzy_match_villages(sample_df, codes_df)
        match_time = time.time() - start
        
        match_count = len(matched_df[matched_df['match_type'] == f'fuzzy_{threshold}'])
        
        return {
            'records_processed': sample_size,
            'matches_found': match_count,
            'match_rate': (match_count / sample_size * 100) if sample_size > 0 else 0,
            'matching_time': match_time,
            'threshold': threshold
        }
    
    def benchmark_complete_pipeline(self, entities_file: str, codes_file: str) -> dict:
        """Benchmark complete mapping pipeline."""
        config = MappingConfig(
            input_entities_file=entities_file,
            input_codes_file=codes_file,
            output_directory=str(self.output_dir / f"test_output_{self.timestamp}"),
            fuzzy_thresholds=[95, 90]
        )
        
        logger = setup_logging(config)
        engine = MappingEngine(config, logger)
        
        start = time.time()
        results, stats = engine.run_complete_mapping()
        pipeline_time = time.time() - start
        
        return {
            'records_processed': stats.total_entities,
            'exact_matches': stats.exact_matches,
            'fuzzy_matches_total': sum(stats.fuzzy_matches.values()),
            'unmatched': stats.unmatched,
            'match_rate': stats.get_match_rate(),
            'pipeline_time': pipeline_time
        }
    
    def benchmark_scalability(self, entities_file: str, codes_file: str, sizes: List[int] = None) -> dict:
        """Benchmark scalability with different dataset sizes."""
        if sizes is None:
            sizes = [100, 500, 1000, 5000]
        
        config = MappingConfig(
            input_entities_file=entities_file,
            input_codes_file=codes_file,
            output_directory=str(self.output_dir)
        )
        
        logger = setup_logging(config)
        loader = DataLoader(config, logger)
        
        entities_df = loader.load_entities()
        codes_df = loader.load_lgd_codes()
        
        scalability_results = []
        
        for size in sizes:
            if size > len(entities_df):
                continue
            
            sample_df = entities_df.head(size).copy()
            
            matcher = ExactMatcher(logger)
            
            start = time.time()
            matched_df = matcher.match_by_uid(sample_df, codes_df)
            duration = time.time() - start
            
            scalability_results.append({
                'size': size,
                'duration': duration,
                'rate': size / duration if duration > 0 else 0
            })
            
            print(f"  Size {size:,}: {duration:.3f}s ({size/duration:.0f} records/s)")
        
        return {
            'records_processed': sum(r['size'] for r in scalability_results),
            'scalability_data': scalability_results
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("LGD MAPPING PERFORMANCE BENCHMARK REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM")
        report_lines.append("")
        
        # Summary statistics
        successful_benchmarks = [r for r in self.results if r.success]
        failed_benchmarks = [r for r in self.results if not r.success]
        
        report_lines.append(f"Total Benchmarks: {len(self.results)}")
        report_lines.append(f"Successful: {len(successful_benchmarks)}")
        report_lines.append(f"Failed: {len(failed_benchmarks)}")
        report_lines.append("")
        
        # Individual benchmark results
        report_lines.append("BENCHMARK RESULTS")
        report_lines.append("-"*80)
        
        for result in self.results:
            report_lines.append(f"\n{result.name}")
            report_lines.append(f"  Status: {'✓ Success' if result.success else '✗ Failed'}")
            
            if result.success:
                report_lines.append(f"  Duration: {result.duration:.3f} seconds")
                report_lines.append(f"  Memory Usage: {result.memory_start:.1f} MB → {result.memory_end:.1f} MB "
                                  f"(+{result.memory_end - result.memory_start:.1f} MB)")
                
                if result.records_processed > 0:
                    report_lines.append(f"  Records Processed: {result.records_processed:,}")
                    report_lines.append(f"  Processing Rate: {result.records_processed / result.duration:.0f} records/second")
                
                if result.metrics:
                    report_lines.append(f"  Additional Metrics:")
                    for key, value in result.metrics.items():
                        if key != 'records_processed' and not key.endswith('_data'):
                            if isinstance(value, float):
                                report_lines.append(f"    {key}: {value:.2f}")
                            else:
                                report_lines.append(f"    {key}: {value}")
            else:
                report_lines.append(f"  Error: {result.error}")
        
        # Performance summary
        if successful_benchmarks:
            report_lines.append("\n" + "="*80)
            report_lines.append("PERFORMANCE SUMMARY")
            report_lines.append("-"*80)
            
            total_duration = sum(r.duration for r in successful_benchmarks)
            total_records = sum(r.records_processed for r in successful_benchmarks)
            avg_memory_growth = np.mean([r.memory_end - r.memory_start for r in successful_benchmarks])
            
            report_lines.append(f"Total Execution Time: {total_duration:.2f} seconds")
            report_lines.append(f"Total Records Processed: {total_records:,}")
            report_lines.append(f"Average Memory Growth: {avg_memory_growth:.1f} MB")
            
            if total_duration > 0:
                report_lines.append(f"Overall Processing Rate: {total_records / total_duration:.0f} records/second")
        
        report_lines.append("\n" + "="*80)
        
        return "\n".join(report_lines)
    
    def save_results(self):
        """Save benchmark results to files."""
        # Save JSON results
        json_file = self.output_dir / f"benchmark_results_{self.timestamp}.json"
        results_data = {
            'timestamp': self.timestamp,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / 1024**3,
                'python_version': sys.version
            },
            'benchmarks': [r.to_dict() for r in self.results]
        }
        
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n✓ JSON results saved to: {json_file}")
        
        # Save text report
        report_file = self.output_dir / f"benchmark_report_{self.timestamp}.txt"
        report = self.generate_report()
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✓ Text report saved to: {report_file}")
        
        return json_file, report_file


def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="LGD Mapping Performance Benchmark")
    parser.add_argument("--entities", required=True, help="Path to entities CSV file")
    parser.add_argument("--codes", required=True, help="Path to LGD codes CSV file")
    parser.add_argument("--output", default="benchmark_results", help="Output directory for results")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (skip some tests)")
    parser.add_argument("--scalability", action="store_true", help="Include scalability tests")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.entities).exists():
        print(f"Error: Entities file not found: {args.entities}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.codes).exists():
        print(f"Error: Codes file not found: {args.codes}", file=sys.stderr)
        sys.exit(1)
    
    print("="*80)
    print("LGD MAPPING PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"Entities file: {args.entities}")
    print(f"Codes file: {args.codes}")
    print(f"Output directory: {args.output}")
    print(f"Mode: {'Quick' if args.quick else 'Comprehensive'}")
    
    benchmark = PerformanceBenchmark(args.output)
    
    # Run benchmarks
    benchmark.run_benchmark(
        "Data Loading",
        benchmark.benchmark_data_loading,
        args.entities,
        args.codes
    )
    
    benchmark.run_benchmark(
        "Exact Matching",
        benchmark.benchmark_exact_matching,
        args.entities,
        args.codes
    )
    
    if not args.quick:
        benchmark.run_benchmark(
            "Fuzzy Matching (95% threshold)",
            benchmark.benchmark_fuzzy_matching,
            args.entities,
            args.codes,
            95
        )
        
        benchmark.run_benchmark(
            "Fuzzy Matching (90% threshold)",
            benchmark.benchmark_fuzzy_matching,
            args.entities,
            args.codes,
            90
        )
    
    benchmark.run_benchmark(
        "Complete Pipeline",
        benchmark.benchmark_complete_pipeline,
        args.entities,
        args.codes
    )
    
    if args.scalability:
        benchmark.run_benchmark(
            "Scalability Test",
            benchmark.benchmark_scalability,
            args.entities,
            args.codes,
            [100, 500, 1000, 5000, 10000]
        )
    
    # Generate and save results
    print("\n" + benchmark.generate_report())
    benchmark.save_results()
    
    # Summary
    successful = sum(1 for r in benchmark.results if r.success)
    total = len(benchmark.results)
    
    print(f"\n{'='*80}")
    print(f"Benchmark Complete: {successful}/{total} tests passed")
    print(f"{'='*80}")
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
