#!/usr/bin/env python3
"""
Main entry point for running screenshot comparison tests
"""
import pytest
import sys
from pathlib import Path

if __name__ == "__main__":
    # Ensure directories exist
    from config import Config

    Config.create_dirs()

    # Run pytest with HTML report
    args = [
        "tests/",
        "-v",
        "--html=reports/pytest_report.html",
        "--self-contained-html",
        "--maxfail=1",
        "-s"
    ]

    sys.exit(pytest.main(args))