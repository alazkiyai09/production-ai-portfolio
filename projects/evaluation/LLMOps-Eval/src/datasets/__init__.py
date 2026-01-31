"""
Test dataset management for LLM evaluation.

This module provides functionality for loading, saving, filtering, and managing
test datasets used for LLM evaluation. Supports YAML and JSON formats.
"""

from src.datasets.dataset_manager import (
    TestCase,
    TestDataset,
    DatasetManager,
    SAMPLE_DATASET_YAML,
    CODE_EVALUATION_DATASET_YAML,
    SAFETY_EVALUATION_DATASET_YAML,
    load_sample_dataset,
    create_test_case,
)

__all__ = [
    "TestCase",
    "TestDataset",
    "DatasetManager",
    "SAMPLE_DATASET_YAML",
    "CODE_EVALUATION_DATASET_YAML",
    "SAFETY_EVALUATION_DATASET_YAML",
    "load_sample_dataset",
    "create_test_case",
]
