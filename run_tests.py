#!/usr/bin/env python3
"""
Test runner script for the Lexora Agentic RAG SDK.

This script provides a convenient way to run tests with various options
and configurations.
"""

import sys
import subprocess
from pathlib import Path


def run_tests(
    test_path: str = "tests/",
    verbose: bool = True,
    coverage: bool = False,
    markers: str = None,
    parallel: bool = False
) -> int:
    """
    Run tests with specified options.
    
    Args:
        test_path: Path to tests to run
        verbose: Whether to run in verbose mode
        coverage: Whether to generate coverage report
        markers: Pytest markers to filter tests (e.g., "unit", "not slow")
        parallel: Whether to run tests in parallel
        
    Returns:
        Exit code from pytest
    """
    cmd = ["python", "-m", "pytest"]
    
    # Add test path
    cmd.append(test_path)
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.extend(["--cov=llm", "--cov=utils", "--cov=models", "--cov=exceptions"])
        cmd.append("--cov-report=html")
        cmd.append("--cov-report=term-missing")
    
    # Add marker filtering
    if markers:
        cmd.extend(["-m", markers])
    
    # Add parallel execution
    if parallel:
        try:
            import pytest_xdist
            cmd.extend(["-n", "auto"])
        except ImportError:
            print("Warning: pytest-xdist not available, running sequentially")
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings"  # Disable warnings for cleaner output
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    """Main entry point for the test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Lexora SDK tests")
    parser.add_argument(
        "path", 
        nargs="?", 
        default="tests/",
        help="Path to tests to run (default: tests/)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run in verbose mode"
    )
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "-m", "--markers",
        help="Run only tests matching given mark expression"
    )
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    parser.add_argument(
        "--unit",
        action="store_const",
        const="unit",
        dest="markers",
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration",
        action="store_const",
        const="integration",
        dest="markers",
        help="Run only integration tests"
    )
    parser.add_argument(
        "--fast",
        action="store_const",
        const="not slow",
        dest="markers",
        help="Run only fast tests (exclude slow tests)"
    )
    
    args = parser.parse_args()
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("Error: pytest is not installed. Install with: pip install pytest")
        return 1
    
    # Run tests
    exit_code = run_tests(
        test_path=args.path,
        verbose=args.verbose,
        coverage=args.coverage,
        markers=args.markers,
        parallel=args.parallel
    )
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())