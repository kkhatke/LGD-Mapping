#!/usr/bin/env python3
"""
Test runner script for LGD Mapping project.

This script provides various options for running tests including:
- All tests
- Specific test categories (unit, integration, performance)
- Coverage reporting
- Performance benchmarking
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"\nExit code: {result.returncode}")
        return result.returncode == 0
    
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def run_unit_tests():
    """Run unit tests only."""
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-v"]
    return run_command(cmd, "Unit Tests")


def run_pytest_tests():
    """Run tests using pytest if available."""
    try:
        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
        return run_command(cmd, "Pytest Tests")
    except FileNotFoundError:
        print("pytest not available, falling back to unittest")
        return run_unit_tests()


def run_coverage_tests():
    """Run tests with coverage reporting."""
    try:
        # Run tests with coverage
        cmd = [sys.executable, "-m", "pytest", "--cov=lgd_mapping", "--cov-report=html", "--cov-report=term-missing", "-v"]
        success = run_command(cmd, "Coverage Tests")
        
        if success:
            print("\nCoverage report generated in htmlcov/index.html")
        
        return success
    except FileNotFoundError:
        print("pytest-cov not available, running basic tests")
        return run_unit_tests()


def run_performance_tests():
    """Run performance tests only."""
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_performance*.py", "-v"]
    return run_command(cmd, "Performance Tests")


def run_integration_tests():
    """Run integration tests only."""
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_integration*.py", "-v"]
    return run_command(cmd, "Integration Tests")


def run_specific_test(test_file):
    """Run a specific test file."""
    cmd = [sys.executable, "-m", "unittest", test_file, "-v"]
    return run_command(cmd, f"Specific Test: {test_file}")


def check_test_environment():
    """Check if the test environment is properly set up."""
    print("Checking test environment...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check required packages
    required_packages = ['pandas', 'numpy', 'rapidfuzz', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    # Check optional test packages
    optional_packages = ['pytest', 'pytest_cov']
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available (optional)")
        except ImportError:
            print(f"- {package} is not available (optional)")
    
    # Check if lgd_mapping package can be imported
    try:
        import lgd_mapping
        print(f"✓ lgd_mapping package is available")
    except ImportError as e:
        print(f"✗ lgd_mapping package import failed: {e}")
        missing_packages.append('lgd_mapping')
    
    if missing_packages:
        print(f"\nMissing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("\n✓ Test environment is ready")
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="LGD Mapping Test Runner")
    parser.add_argument("--type", choices=["all", "unit", "integration", "performance", "coverage"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--file", help="Run specific test file")
    parser.add_argument("--check-env", action="store_true", help="Check test environment")
    parser.add_argument("--use-pytest", action="store_true", help="Use pytest instead of unittest")
    
    args = parser.parse_args()
    
    if args.check_env:
        return 0 if check_test_environment() else 1
    
    # Check environment first
    if not check_test_environment():
        print("\nTest environment check failed. Please fix the issues above.")
        return 1
    
    success = True
    
    if args.file:
        success = run_specific_test(args.file)
    elif args.type == "unit":
        success = run_unit_tests()
    elif args.type == "integration":
        success = run_integration_tests()
    elif args.type == "performance":
        success = run_performance_tests()
    elif args.type == "coverage":
        success = run_coverage_tests()
    elif args.type == "all":
        if args.use_pytest:
            success = run_pytest_tests()
        else:
            success = run_unit_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())