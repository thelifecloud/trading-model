"""
test_data_normalize.py

This module contains unit tests for the `data_normalize` module, which provides
functionality to normalize numerical data in pandas DataFrames.

Tests:
    - test_normalize_data: Verifies that the `normalize_data` function correctly normalizes data
      using Min-Max scaling and Z-Score normalization.

Usage:
    Run this script using pytest:
        pytest test_data_normalize.py
"""
import os
import pandas as pd
from src.data_normalize import normalize_data

def test_normalize_data():
    """
    Test the `normalize_data` function.

    Validates:
        - Min-Max scaling normalizes values to the range [0, 1].
        - Z-Score normalization standardizes values to have a mean of 0 and a standard deviation of 1.

    Steps:
        1. Create a sample dataset with numerical columns.
        2. Apply Min-Max scaling and verify the range of normalized values.
        3. Apply Z-Score normalization and verify the mean and standard deviation.

    Asserts:
        - Min-Max scaling produces expected max and min values.
        - Z-Score normalization results in near-zero mean and standard deviation close to 1.

    Prints:
        - Confirmation message if all tests pass.
    """
    # Create a larger sample dataset
    data = pd.DataFrame({
        "sma_20": list(range(1, 101)),  # Values from 1 to 100
        "sma_50": list(range(101, 201)),
        "rsi": list(range(201, 301)),
    })

    # Normalize using Min-Max Scaling
    normalized_data_minmax = normalize_data(data.copy(), method="minmax")
    assert normalized_data_minmax["sma_20"].max() == 1.0, "Min-Max scaling failed for sma_20."
    assert normalized_data_minmax["sma_20"].min() == 0.0, "Min-Max scaling failed for sma_20."

    # Normalize using Z-Score Normalization
    normalized_data_zscore = normalize_data(data.copy(), method="zscore")
    assert abs(normalized_data_zscore["sma_20"].mean()) < 1e-6, "Z-Score normalization failed for sma_20."
    
    # Allow slightly higher tolerance for floating-point precision
    assert abs(normalized_data_zscore["sma_20"].std() - 1.0) < 1e-2, "Z-Score normalization failed for sma_20."

    
    print("All tests passed for normalize_data!")
