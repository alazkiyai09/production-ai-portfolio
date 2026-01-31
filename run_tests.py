#!/usr/bin/env python
"""
Test runner script for AgenticFlow.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run unit tests only
    python run_tests.py --coverage   # Run with coverage report
    python run_tests.py --verbose    # Verbose output
"""

import argparse
import subprocess
import sys


def run_tests(
    unit_only: bool = False,
    integration_only: bool = False,
    coverage: bool = False,
    verbose: bool = False,
    marker: str = None,
    pattern: str = None,
):
    """
    Run the test suite with specified options.

    Args:
        unit_only: Run only unit tests
        integration_only: Run only integration tests
        coverage: Generate coverage report
        verbose: Verbose output
        marker: Run tests with specific marker
        pattern: Run tests matching pattern
    """
    # Build pytest command
    cmd = ["pytest", "tests/"]

    # Add verbosity
    if verbose:
        cmd.append("-v")

    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
        ])

    # Add markers
    if unit_only:
        cmd.extend(["-m", "unit"])
    elif integration_only:
        cmd.extend(["-m", "integration"])
    elif marker:
        cmd.extend(["-m", marker])

    # Add pattern
    if pattern:
        cmd.extend(["-k", pattern])

    # Add other options
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "-ra",  # Show summary of all test results
    ])

    # Print command
    print(f"Running: {' '.join(cmd)}\n")

    # Run tests
    result = subprocess.run(cmd)

    # Print coverage info if generated
    if coverage and result.returncode == 0:
        print("\n" + "=" * 60)
        print("Coverage report generated: htmlcov/index.html")
        print("=" * 60)

    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for AgenticFlow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                 Run all tests
  python run_tests.py --unit          Run unit tests only
  python run_tests.py --integration   Run integration tests only
  python run_tests.py --coverage      Run with coverage report
  python run_tests.py -v              Verbose output
  python run_tests.py -k "calculator" Run tests matching pattern
  python run_tests.py -m "slow"       Run tests with 'slow' marker
        """,
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only",
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--marker", "-m",
        type=str,
        help="Run tests with specific marker",
    )
    parser.add_argument(
        "--pattern", "-k",
        type=str,
        help="Run tests matching pattern",
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.unit and args.integration:
        print("Error: --unit and --integration are mutually exclusive")
        return 1

    # Run tests
    return run_tests(
        unit_only=args.unit,
        integration_only=args.integration,
        coverage=args.coverage,
        verbose=args.verbose,
        marker=args.marker,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    sys.exit(main())
