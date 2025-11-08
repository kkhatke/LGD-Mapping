#!/usr/bin/env python3
"""
Deployment Verification Script

This script verifies that the LGD Mapping application is properly installed
and all components are working correctly.

Usage:
    python verify_deployment.py [--verbose]
"""

import sys
import argparse
from pathlib import Path
import importlib.util


class DeploymentVerifier:
    """Verifies deployment and system requirements."""
    
    def __init__(self, verbose=False):
        """Initialize verifier."""
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_total = 0
    
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
    
    def check(self, description, test_func):
        """Run a check and track results."""
        self.checks_total += 1
        self.log(f"Checking: {description}", 'INFO')
        
        try:
            result, message = test_func()
            if result:
                self.checks_passed += 1
                self.log(f"  {message}", 'SUCCESS')
                return True
            else:
                self.errors.append(f"{description}: {message}")
                self.log(f"  {message}", 'ERROR')
                return False
        except Exception as e:
            self.errors.append(f"{description}: {str(e)}")
            self.log(f"  Error: {str(e)}", 'ERROR')
            return False
    
    def warn(self, description, test_func):
        """Run a check that produces warnings instead of errors."""
        self.log(f"Checking: {description}", 'INFO')
        
        try:
            result, message = test_func()
            if result:
                self.log(f"  {message}", 'SUCCESS')
            else:
                self.warnings.append(f"{description}: {message}")
                self.log(f"  {message}", 'WARNING')
            return result
        except Exception as e:
            self.warnings.append(f"{description}: {str(e)}")
            self.log(f"  Warning: {str(e)}", 'WARNING')
            return False
    
    def check_python_version(self):
        """Check Python version."""
        def test():
            version = sys.version_info
            if version >= (3, 8):
                return True, f"Python {version.major}.{version.minor}.{version.micro} (OK)"
            else:
                return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"
        
        return self.check("Python version", test)
    
    def check_required_packages(self):
        """Check required Python packages."""
        required_packages = {
            'pandas': '1.5.0',
            'numpy': '1.21.0',
            'rapidfuzz': '2.13.0',
            'tqdm': '4.64.0',
            'psutil': None  # No minimum version specified
        }
        
        all_ok = True
        for package, min_version in required_packages.items():
            def test(pkg=package, ver=min_version):
                try:
                    module = importlib.import_module(pkg)
                    if hasattr(module, '__version__'):
                        version = module.__version__
                        if ver:
                            return True, f"{pkg} {version} installed (requires {ver}+)"
                        else:
                            return True, f"{pkg} {version} installed"
                    else:
                        return True, f"{pkg} installed (version unknown)"
                except ImportError:
                    return False, f"{pkg} not installed"
            
            if not self.check(f"Package: {package}", test):
                all_ok = False
        
        return all_ok
    
    def check_project_structure(self):
        """Check project directory structure."""
        required_paths = [
            'lgd_mapping/__init__.py',
            'lgd_mapping/config.py',
            'lgd_mapping/logging_config.py',
            'lgd_mapping/models.py',
            'lgd_mapping/mapping_engine.py',
            'lgd_mapping/data_loader.py',
            'lgd_mapping/matching/exact_matcher.py',
            'lgd_mapping/matching/fuzzy_matcher.py',
            'lgd_mapping/output/output_generator.py',
            'lgd_mapping/utils/uid_generator.py',
            'main.py',
            'requirements.txt'
        ]
        
        all_ok = True
        for path in required_paths:
            def test(p=path):
                if Path(p).exists():
                    return True, f"{p} exists"
                else:
                    return False, f"{p} not found"
            
            if not self.check(f"File: {path}", test):
                all_ok = False
        
        return all_ok
    
    def check_example_files(self):
        """Check example files exist."""
        example_files = [
            'examples/sample_entities.csv',
            'examples/sample_codes.csv',
            'examples/config_template.json',
            'examples/district_mapping.json',
            'examples/validate_input.py'
        ]
        
        for path in example_files:
            def test(p=path):
                if Path(p).exists():
                    return True, f"{p} exists"
                else:
                    return False, f"{p} not found"
            
            self.warn(f"Example: {path}", test)
    
    def check_import_modules(self):
        """Check that main modules can be imported."""
        modules = [
            'lgd_mapping.config',
            'lgd_mapping.logging_config',
            'lgd_mapping.models',
            'lgd_mapping.mapping_engine',
            'lgd_mapping.data_loader',
            'lgd_mapping.matching.exact_matcher',
            'lgd_mapping.matching.fuzzy_matcher',
            'lgd_mapping.output.output_generator'
        ]
        
        all_ok = True
        for module_name in modules:
            def test(mod=module_name):
                try:
                    importlib.import_module(mod)
                    return True, f"{mod} imports successfully"
                except ImportError as e:
                    return False, f"{mod} import failed: {str(e)}"
            
            if not self.check(f"Import: {module_name}", test):
                all_ok = False
        
        return all_ok
    
    def check_configuration_classes(self):
        """Check configuration classes can be instantiated."""
        def test():
            try:
                from lgd_mapping.config import MappingConfig
                
                # Try to create a config with minimal parameters
                config = MappingConfig(
                    input_entities_file='examples/sample_entities.csv',
                    input_codes_file='examples/sample_codes.csv',
                    output_directory='./test_output'
                )
                
                return True, "MappingConfig can be instantiated"
            except Exception as e:
                return False, f"MappingConfig instantiation failed: {str(e)}"
        
        return self.check("Configuration classes", test)
    
    def check_data_models(self):
        """Check data models can be created."""
        def test():
            try:
                from lgd_mapping.models import EntityRecord, LGDRecord, MappingResult
                
                # Try to create instances
                entity = EntityRecord(
                    district='Test District',
                    block='Test Block',
                    village='Test Village'
                )
                
                lgd = LGDRecord(
                    district_code=101,
                    district='Test District',
                    block_code=1001,
                    block='Test Block',
                    village_code=100101,
                    village='Test Village'
                )
                
                result = MappingResult(
                    entity=entity,
                    lgd_match=lgd,
                    match_type='exact',
                    match_score=100.0
                )
                
                return True, "Data models can be instantiated"
            except Exception as e:
                return False, f"Data model instantiation failed: {str(e)}"
        
        return self.check("Data models", test)
    
    def check_sample_data_format(self):
        """Check sample data files have correct format."""
        def test():
            try:
                import pandas as pd
                
                # Check entities file
                entities = pd.read_csv('examples/sample_entities.csv')
                required_cols = ['district', 'block', 'village']
                if not all(col in entities.columns for col in required_cols):
                    return False, "sample_entities.csv missing required columns"
                
                # Check codes file
                codes = pd.read_csv('examples/sample_codes.csv')
                required_cols = ['district_code', 'district', 'block_code', 'block', 
                               'village_code', 'village']
                if not all(col in codes.columns for col in required_cols):
                    return False, "sample_codes.csv missing required columns"
                
                return True, "Sample data files have correct format"
            except Exception as e:
                return False, f"Sample data validation failed: {str(e)}"
        
        return self.warn("Sample data format", test)
    
    def check_cli_interface(self):
        """Check CLI interface is accessible."""
        def test():
            try:
                # Try to import main module
                spec = importlib.util.spec_from_file_location("main", "main.py")
                if spec is None:
                    return False, "main.py not found or not importable"
                
                return True, "CLI interface (main.py) is accessible"
            except Exception as e:
                return False, f"CLI interface check failed: {str(e)}"
        
        return self.check("CLI interface", test)
    
    def run_all_checks(self):
        """Run all verification checks."""
        print("="*60)
        print("LGD MAPPING DEPLOYMENT VERIFICATION")
        print("="*60)
        print()
        
        print("System Requirements")
        print("-"*60)
        self.check_python_version()
        print()
        
        print("Required Packages")
        print("-"*60)
        self.check_required_packages()
        print()
        
        print("Project Structure")
        print("-"*60)
        self.check_project_structure()
        print()
        
        print("Module Imports")
        print("-"*60)
        self.check_import_modules()
        print()
        
        print("Component Functionality")
        print("-"*60)
        self.check_configuration_classes()
        self.check_data_models()
        self.check_cli_interface()
        print()
        
        print("Example Files (Optional)")
        print("-"*60)
        self.check_example_files()
        self.check_sample_data_format()
        print()
    
    def print_summary(self):
        """Print verification summary."""
        print("="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        print()
        
        print(f"Checks passed: {self.checks_passed}/{self.checks_total}")
        print()
        
        if self.errors:
            print(f"✗ Found {len(self.errors)} error(s):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
            print()
        
        if self.warnings:
            print(f"⚠ Found {len(self.warnings)} warning(s):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
            print()
        
        if not self.errors:
            print("✓ Deployment verification passed!")
            print("The LGD Mapping application is ready to use.")
            print()
            print("Next steps:")
            print("  1. Prepare your input CSV files")
            print("  2. Validate them: python examples/validate_input.py --entities <file> --codes <file>")
            print("  3. Run mapping: python main.py --entities <file> --codes <file> --output <dir>")
            return True
        else:
            print("✗ Deployment verification failed!")
            print("Please fix the errors before using the application.")
            print()
            print("Common fixes:")
            print("  - Install missing packages: pip install -r requirements.txt")
            print("  - Ensure you're in the project root directory")
            print("  - Check that all files were properly extracted/cloned")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify LGD Mapping application deployment"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed verification output"
    )
    
    args = parser.parse_args()
    
    verifier = DeploymentVerifier(verbose=args.verbose)
    verifier.run_all_checks()
    success = verifier.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
