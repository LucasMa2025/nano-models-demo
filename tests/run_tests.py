"""
Test runner for Nano Models framework.

Usage:
    python run_tests.py           # Run all tests
    python run_tests.py -v        # Run with verbose output
    python run_tests.py models    # Run specific test module
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_all_tests(verbosity: int = 2):
    """Run all unit tests."""
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.dirname(__file__),
        pattern='test_*.py'
    )
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_tests(module_name: str, verbosity: int = 2):
    """Run tests from a specific module."""
    loader = unittest.TestLoader()
    
    try:
        suite = loader.loadTestsFromName(f'test_{module_name}')
    except ModuleNotFoundError:
        print(f"Test module 'test_{module_name}' not found.")
        return False
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main entry point."""
    verbosity = 2
    
    if '-v' in sys.argv:
        verbosity = 3
        sys.argv.remove('-v')
    
    if len(sys.argv) > 1:
        # Run specific module
        module_name = sys.argv[1]
        success = run_specific_tests(module_name, verbosity)
    else:
        # Run all tests
        success = run_all_tests(verbosity)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
