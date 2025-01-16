"""
test_data_verifier.py

This module contains unit tests for the `data_verifier` module, which provides
utilities for validating data integrity in pandas DataFrames.

Tests:
    - test_check_missing_values: Verifies the function correctly identifies missing values.
    - test_check_column_types: Verifies the function detects mismatched column data types.
    - test_verify_dataset: Verifies comprehensive dataset validation, including missing values and type mismatches.

Fixtures:
    - example_dataframe: Provides a sample pandas DataFrame for testing purposes.

Usage:
    Run this script using pytest:
        pytest test_data_verifier.py
"""
import sys
import os
import pytest
import pandas as pd

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the verifier functions
from src.data_verifier import check_missing_values, check_column_types, verify_dataset

@pytest.fixture
def example_dataframe():
    """
    Provides a sample DataFrame for testing.

    Returns:
        pd.DataFrame: A DataFrame with missing values for testing purposes.
    """
    return pd.DataFrame({
        "column1": [1, 2, None],
        "column2": [4, None, 6],
    })

def test_check_missing_values(example_dataframe):
    """
    Test the `check_missing_values` function.

    Validates:
        - Correct identification of total missing values.
        - Accurate summary of missing values by column.

    Args:
        example_dataframe (pd.DataFrame): Fixture providing the sample DataFrame.

    Asserts:
        - The returned dictionary matches the expected result.
    """
    result = check_missing_values(example_dataframe)
    expected_result = {
        "total_missing": 2,
        "missing_by_column": {"column1": 1, "column2": 1},
    }
    assert result == expected_result, f"Unexpected result: {result}"

def test_check_column_types(example_dataframe):
    """
    Test the `check_column_types` function.

    Validates:
        - Columns match the expected data types.

    Args:
        example_dataframe (pd.DataFrame): Fixture providing the sample DataFrame.

    Asserts:
        - The result is an empty list (indicating no type mismatches).
    """
    # Define expected types
    expected_types = {"column1": "float64", "column2": "float64"}
    result = check_column_types(example_dataframe, expected_types)
    assert result == [], f"Mismatched types found: {result}"

def test_verify_dataset(example_dataframe):
    """
    Test the `verify_dataset` function.

    Validates:
        - Correct identification of missing values.
        - Accurate detection of type mismatches (or lack thereof).

    Args:
        example_dataframe (pd.DataFrame): Fixture providing the sample DataFrame.

    Asserts:
        - The "missing_values" key contains the correct summary of missing data.
        - The "type_mismatches" key is an empty list (indicating no mismatches).
    """
    # Define expected types
    expected_types = {"column1": "float64", "column2": "float64"}
    result = verify_dataset(example_dataframe, expected_types)
    
    # Check for missing values
    assert result["missing_values"]["total_missing"] == 2, "Total missing values mismatch."
    assert result["missing_values"]["missing_by_column"] == {"column1": 1, "column2": 1}, "Missing by column mismatch."
    
    # Check for type mismatches
    assert result["type_mismatches"] == [], f"Unexpected type mismatches: {result['type_mismatches']}"
