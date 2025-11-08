#!/usr/bin/env python3
"""
Performance Benchmarking Script

This script benchmarks the LGD Mapping application performance
with different dataset sizes and configurations.

Usage:
    python benchmark_performance.py [--quick] [--output <dir>]
"""

import argparse
import sys
import time
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import psutil


def generate_test_data(num_entities, num_codes, output_dir):
    """Generate synthetic test data."""
    print(f"Generating test data: {num_entities} entities, {num_codes} codes...")
    
    # Generate entities
    districts = [f"District_{i%10}" for i in range(num_entities)]
    blocks = [f"Block_{i%50}" for i in range(num_entities)]
    villages = [f"Village_{i}" for i in range(num_entities)]
    
    entities_df = pd.DataFrame({
        'district': districts,
        'block': blocks,
        'village': villages
    })
    
    entities_file = Path(output_dir) / 'test_entities.csv'
    entities_df.to_csv(entities_file, index=False)
    
    # Generate codes (with some matches)
    district_codes = [100 + (i%10) for i in range(num_codes)]
    districts_lgd = [f"District_{i%10}" for i in range(num_codes)]
    block_codes = [1000 + (i%50) for i in range(num_codes)]
    blocks_lgd = [f"Block_{i%50}" for i in range(num_codes)]
    village_codes = [10000 + i for i in range(num_codes)]
    villages_lgd = [f"Village_{i}" for i in range(num_codes)]
    
    codes_df = pd.DataFrame({
        'district_code': district_codes,
        'district': districts_lgd,
        'block_code': block_codes,
        'block': blocks_lgd,
        'village_code': village_codes,
        'village': villages_lgd
    })
    
    codes_file = Path(output_dir) / 'test_codes.csv'
    codes_df.to_csv(codes_file, index=False)
    
    return str(entities_file), str(codes_file)


def run_benchmark(entities_file, codes_file, output_dir, config):
    """Run a single benchmark test."""
    from lgd_mapping.config import MappingConfig
    from lgd_mapping.logging_config import setup_logging
    from lgd_mapping.mapping_engine import MappingEngine
    
    # Create configuration
    mapping_config = MappingConfig(
        input_entities_file=entities_file,
        input_codes_file=codes_file,
        output_directory=output_dir,
        fuzzy_thresholds=config.get('thresholds', [95, 90]),
        chunk_size=config.get('chunk_size'),
        log_level='WARNING'  # Reduce logging overhead
    )
    
    # Set up logging
    logger = setup_logging(mapping_config)
    
    # Initialize engine
    engine = MappingEngine(mapping_config, logger)
    
    # Track memory before
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run mapping
    start_time = time.time()
    results, stats = engine.run_complete_mapping()
    end_time = time.time()
    
    # Track memory after
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    duration = end_time - start_time
    memory_used = memory_after - memory_before
    entities_per_second = stats.total_entities / duration if duration > 0 else 0
    
    return {
        'duration': duration,
        'memory_used': memory_used,
        'memory_peak': memory_after,
        'entities_per_second': entities_per_second,
        'total_entities': stats.total_entities,
        'exact_matches': stats.exact_matches,
        'fuzzy_matches': sum(stats.fuzzy_matches.values()),
        'unmatched': stats.unmatched,
        'match_rate': stats.get_match_rate()
    }


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_memory(mb):
    """Format memory in human-readable format."""
    if mb < 1024:
        return f"{mb:.1f} MB"
    else:
        gb = mb / 1024
        return f"{gb:.2f} GB"


def print_benchmark_result(name, config, result):
    """Print benchmark result."""
    print(f"\n{name}")
    print("-" * 60)
    print(f"Configuration:")
    print(f"  Entities: {result['total_entities']:,}")
    print(f"  Thresholds: {config.get('thresholds', [95, 90])}")
    print(f"  Chunk size: {config.get('chunk_size', 'Auto')}")
    print()
    print(f"Performance:")
    print(f"  Duration: {format_duration(result['duration'])}")
    print(f"  Processing rate: {result['entities_per_second']:.0f} entities/second")
    print(f"  Memory used: {format_memory(result['memory_used'])}")
    print(f"  Peak memory: {format_memory(result['memory_peak'])}")
    print()
    print(f"Results:")
    print(f"  Exact matches: {result['exact_matches']:,} ({result['exact_matches']/result['total_entities']*100:.1f}%)")
    print(f"  Fuzzy matches: {result['fuzzy_matches']:,} ({result['fuzzy_matches']/result['total_entities']*100:.1f}%)")
    print(f"  Unmatched: {result['unmatched']:,} ({result['unmatched']/result['total_entities']*100:.1f}%)")
    print(f"  Overall match rate: {result['match_rate']:.1f}%")


def run_quick_benchmark(temp_dir):
    """Run quick benchmark with small dataset."""
    print("="*60)
    print("QUICK PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Small dataset test
    entities_file, codes_file = generate_test_data(1000, 5000, temp_dir)
    output_dir = Path(temp_dir) / 'output_small'
    output_dir.mkdir(exist_ok=True)
    
    config = {'thresholds': [95, 90], 'chunk_size': None}
    result = run_benchmark(entities_file, codes_file, str(output_dir), config)
    print_benchmark_result("Small Dataset (1,000 entities)", config, result)


def run_full_benchmark(temp_dir):
    """Run comprehensive benchmark suite."""
    print("="*60)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("="*60)
    
    benchmarks = [
        {
            'name': 'Small Dataset',
            'entities': 1000,
            'codes': 5000,
            'config': {'thresholds': [95, 90], 'chunk_size': None}
        },
        {
            'name': 'Medium Dataset',
            'entities': 10000,
            'codes': 50000,
            'config': {'thresholds': [95, 90], 'chunk_size': None}
        },
        {
            'name': 'Large Dataset (No Chunking)',
            'entities': 50000,
            'codes': 100000,
            'config': {'thresholds': [95, 90], 'chunk_size': None}
        },
        {
            'name': 'Large Dataset (With Chunking)',
            'entities': 50000,
            'codes': 100000,
            'config': {'thresholds': [95, 90], 'chunk_size': 5000}
        },
        {
            'name': 'Single Threshold (Speed Optimized)',
            'entities': 10000,
            'codes': 50000,
            'config': {'thresholds': [95], 'chunk_size': None}
        },
        {
            'name': 'Multiple Thresholds (Accuracy Optimized)',
            'entities': 10000,
            'codes': 50000,
            'config': {'thresholds': [98, 95, 92, 90], 'chunk_size': None}
        }
    ]
    
    results = []
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] Running: {benchmark['name']}")
        
        # Generate test data
        entities_file, codes_file = generate_test_data(
            benchmark['entities'],
            benchmark['codes'],
            temp_dir
        )
        
        # Create output directory
        output_dir = Path(temp_dir) / f'output_{i}'
        output_dir.mkdir(exist_ok=True)
        
        # Run benchmark
        try:
            result = run_benchmark(
                entities_file,
                codes_file,
                str(output_dir),
                benchmark['config']
            )
            results.append((benchmark['name'], benchmark['config'], result))
            print_benchmark_result(benchmark['name'], benchmark['config'], result)
        except Exception as e:
            print(f"✗ Benchmark failed: {e}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print("\nProcessing Rates:")
    for name, config, result in results:
        print(f"  {name:40s}: {result['entities_per_second']:>8.0f} entities/sec")
    
    print("\nMemory Usage:")
    for name, config, result in results:
        print(f"  {name:40s}: {format_memory(result['memory_peak']):>12s}")
    
    print("\nMatch Rates:")
    for name, config, result in results:
        print(f"  {name:40s}: {result['match_rate']:>6.1f}%")


def check_system_resources():
    """Check and display system resources."""
    print("System Resources:")
    print("-" * 60)
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    print(f"CPU cores: {cpu_count} physical, {cpu_count_logical} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    memory_total_gb = memory.total / (1024**3)
    memory_available_gb = memory.available / (1024**3)
    print(f"Memory: {memory_available_gb:.1f} GB available / {memory_total_gb:.1f} GB total")
    
    # Disk info
    disk = psutil.disk_usage('.')
    disk_free_gb = disk.free / (1024**3)
    print(f"Disk space: {disk_free_gb:.1f} GB free")
    
    print()
    
    # Check if resources are sufficient
    warnings = []
    if memory_available_gb < 2:
        warnings.append("Low available memory (<2 GB). Large dataset benchmarks may fail.")
    if disk_free_gb < 1:
        warnings.append("Low disk space (<1 GB). Benchmarks may fail.")
    
    if warnings:
        print("⚠ Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark LGD Mapping application performance"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with small dataset only"
    )
    
    parser.add_argument(
        "--output",
        help="Output directory for benchmark results (default: temp directory)"
    )
    
    args = parser.parse_args()
    
    # Check system resources
    check_system_resources()
    
    # Create temporary directory for test data
    if args.output:
        temp_dir = args.output
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        temp_dir = tempfile.mkdtemp(prefix='lgd_benchmark_')
        cleanup = True
    
    try:
        print(f"Using directory: {temp_dir}\n")
        
        if args.quick:
            run_quick_benchmark(temp_dir)
        else:
            print("Running comprehensive benchmark suite...")
            print("This may take several minutes depending on your system.\n")
            run_full_benchmark(temp_dir)
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
        
        if not args.output:
            print(f"\nTemporary files will be cleaned up.")
        else:
            print(f"\nBenchmark data saved to: {temp_dir}")
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nBenchmark failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup temporary directory if not specified
        if cleanup and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
