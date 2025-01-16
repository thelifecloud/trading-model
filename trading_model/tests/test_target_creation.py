"""
test_target_creation.py

This module contains unit tests for the `target_creation` module, specifically
testing the `add_target` function for correct binary target creation.

Tests:
    - test_add_target: Ensures that the `add_target` function correctly assigns binary
      target values based on specified indicator confluence conditions.

Usage:
    Run this script using pytest:
        pytest test_target_creation.py
"""

import os
import pandas as pd
from src.target_creation import add_target

def test_add_target():
    """
    Test the `add_target` function for correct target creation.

    Validates:
        - Targets are assigned correctly based on the defined confluence logic:
            - 'target' is 1 if SMA(20) > SMA(50) and MACD > Signal Line.
            - 'target' is 0 otherwise.

    Steps:
        1. Create a small sample dataset with SMA, MACD, and Signal Line columns.
        2. Apply the `add_target` function to generate the 'target' column.
        3. Verify that the 'target' column matches the expected values.

    Asserts:
        - The 'target' column matches the expected list of binary values.

    Prints:
        - Confirmation message if all tests pass.
    """
    # Create a small dataset for testing
    data = pd.DataFrame({
        "sma_20": [1.0, 2.0, 3.0, 4.0, 5.0],
        "sma_50": [5.0, 4.0, 3.0, 2.0, 1.0],
        "macd": [1.0, 2.0, 3.0, 4.0, 5.0],
        "signal_line": [5.0, 4.0, 3.0, 2.0, 1.0],
    })

    # Apply the function
    data_with_target = add_target(data)

    # Correct expected targets
    expected_targets = [0, 0, 0, 1, 1]  # Correct based on the logic

    # Check that the target column matches expected values
    assert (data_with_target["target"].tolist() == expected_targets), \
        f"Target column does not match expected values: {data_with_target['target'].tolist()}"

    print("All tests passed for add_target!")

