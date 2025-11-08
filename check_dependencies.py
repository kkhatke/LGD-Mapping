#!/usr/bin/env python3
"""
Dependency Checker Script

This script checks Python package dependencies and their versions
for the LGD Mapping application.

Usage:
    python check_dependencies.py [--fix]
"""

import sys
import argparse
import subprocess
import importlib
from packaging import version as pkg_version


class DependencyChecker:
    """Check and validate package dependencies."""
    
    # Required dependencies with minimum versions
    REQUIRED_DEPS = {
        'pandas': '1.5.0',
        'numpy': '1.21.0',
        'rapidfuzz': '2.13.0',
        'tqdm': '4.64.0',
    }
    
    # Optional dependencies for enhanced functionality
    OPTIONAL_DEPS = {
        'psutil': None,  # For performance monitoring
        'pydantic': '1.10.0',  # For enhanced data validation
    }
    
    # Development dependencies
    DEV_DEPS = {
        'pytest': '7.0.0',
        'pytest-cov': '4.0.0',
        'black': '22.0.0',
        'flake8': '5.0.0',
        'mypy': '0.991',
    }
    
    def __init__(self, verbose=False):
        """Initialize checker."""
        self.verbose = verbose
        self.missing_required = []
        self.missing_optional = []
        self.version_issues = []
    
    def log(self, message, level='INFO'):
        """Log a message."""
        if self.verbose or level in ['ERROR', 'WARNING', 'SUCCESS']:
            prefix = {
                'INFO': 'ℹ',
                'SUCCESS': '✓',
                'WARNING': '⚠',
                'ERROR': '✗'
            }.get(level, ' ')
            print(f"{prefix} {message}")
    
    def check_package(self, package_name, min_version=None):
        """Check if a package is installed and meets version requirements."""
        try:
            module = importlib.import_module(package_name)
            
            if hasattr(module, '__version__'):
                installed_version = module.__version__
                
                if min_version:
                    try:
                        if pkg_version.parse(installed_version) >= pkg_version.parse(min_version):
                            self.log(
                                f"{package_name} {installed_version} (requires {min_version}+)",
                                'SUCCESS'
                            )
                            return True, installed_version
                        else:
                            self.log(
                                f"{package_name} {installed_version} (requires {min_version}+)",
                                'WARNING'
                            )
                            return False, installed_version
                    except Exception:
                        # Version parsing failed, assume OK
                        self.log(
                            f"{package_name} {installed_version} (version check skipped)",
                            'SUCCESS'
                        )
                        return True, installed_version
                else:
                    self.log(f"{package_name} {installed_version}", 'SUCCESS')
                    return True, installed_version
            else:
                self.log(f"{package_name} installed (version unknown)", 'SUCCESS')
                return True, 'unknown'
                
        except ImportError:
            self.log(f"{package_name} not installed", 'ERROR')
            return False, None
    
    def check_required_dependencies(self):
        """Check all required dependencies."""
        print("Required Dependencies:")
        print("-" * 60)
        
        all_ok = True
        for package, min_version in self.REQUIRED_DEPS.items():
            ok, installed_version = self.check_package(package, min_version)
            
            if not ok:
                if installed_version is None:
                    self.missing_required.append(package)
                else:
                    self.version_issues.append((package, installed_version, min_version))
                all_ok = False
        
        print()
        return all_ok
    
    def check_optional_dependencies(self):
        """Check optional dependencies."""
        print("Optional Dependencies:")
        print("-" * 60)
        
        for package, min_version in self.OPTIONAL_DEPS.items():
            ok, installed_version = self.check_package(package, min_version)
            
            if not ok:
                if installed_version is None:
                    self.missing_optional.append(package)
                    self.log(
                        f"  Note: {package} provides enhanced functionality but is not required",
                        'INFO'
                    )
        
        print()
    
    def check_dev_dependencies(self):
        """Check development dependencies."""
        print("Development Dependencies (Optional):")
        print("-" * 60)
        
        for package, min_version in self.DEV_DEPS.items():
            self.check_package(package, min_version)
        
        print()
    
    def print_summary(self):
        """Print dependency check summary."""
        print("="*60)
        print("DEPENDENCY CHECK SUMMARY")
        print("="*60)
        print()
        
        if self.missing_required:
            print(f"✗ Missing required packages ({len(self.missing_required)}):")
            for package in self.missing_required:
                min_version = self.REQUIRED_DEPS[package]
                print(f"  - {package}>={min_version}")
            print()
        
        if self.version_issues:
            print(f"⚠ Version issues ({len(self.version_issues)}):")
            for package, installed, required in self.version_issues:
                print(f"  - {package}: {installed} installed, {required}+ required")
            print()
        
        if self.missing_optional:
            print(f"ℹ Missing optional packages ({len(self.missing_optional)}):")
            for package in self.missing_optional:
                print(f"  - {package} (provides enhanced functionality)")
            print()
        
        if not self.missing_required and not self.version_issues:
            print("✓ All required dependencies are satisfied!")
            print()
            
            if self.missing_optional:
                print("Optional packages are missing but the application will work.")
                print("Install them for enhanced functionality:")
                print(f"  pip install {' '.join(self.missing_optional)}")
                print()
            
            return True
        else:
            print("✗ Dependency check failed!")
            print()
            print("To fix, run:")
            print("  pip install -r requirements.txt")
            print()
            
            if self.missing_required:
                print("Or install missing packages individually:")
                for package in self.missing_required:
                    min_version = self.REQUIRED_DEPS[package]
                    print(f"  pip install {package}>={min_version}")
            
            if self.version_issues:
                print("\nOr upgrade packages with version issues:")
                for package, _, required in self.version_issues:
                    print(f"  pip install --upgrade {package}>={required}")
            
            print()
            return False
    
    def fix_dependencies(self):
        """Attempt to install missing dependencies."""
        print("Attempting to fix dependencies...")
        print()
        
        packages_to_install = []
        
        for package in self.missing_required:
            min_version = self.REQUIRED_DEPS[package]
            packages_to_install.append(f"{package}>={min_version}")
        
        for package, _, required in self.version_issues:
            packages_to_install.append(f"{package}>={required}")
        
        if not packages_to_install:
            print("No packages need to be installed.")
            return True
        
        print(f"Installing: {', '.join(packages_to_install)}")
        print()
        
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + packages_to_install
            )
            print()
            print("✓ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print()
            print(f"✗ Installation failed: {e}")
            return False


def check_python_version():
    """Check Python version."""
    print("Python Version:")
    print("-" * 60)
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version >= (3, 8):
        print(f"✓ Python {version_str} (OK)")
        print()
        return True
    else:
        print(f"✗ Python {version_str} (requires 3.8+)")
        print()
        print("Please upgrade Python to version 3.8 or higher.")
        print()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check Python package dependencies for LGD Mapping application"
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to install missing dependencies"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Also check development dependencies"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("LGD MAPPING DEPENDENCY CHECK")
    print("="*60)
    print()
    
    # Check Python version first
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    checker = DependencyChecker(verbose=args.verbose)
    
    required_ok = checker.check_required_dependencies()
    checker.check_optional_dependencies()
    
    if args.dev:
        checker.check_dev_dependencies()
    
    # Print summary
    success = checker.print_summary()
    
    # Attempt to fix if requested
    if args.fix and not success:
        if checker.fix_dependencies():
            print("Please run the dependency check again to verify:")
            print("  python check_dependencies.py")
            print()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
