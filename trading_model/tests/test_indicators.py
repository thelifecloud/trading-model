"""
test_indicators.py

This module contains unit tests for the `indicators` module, specifically testing
the `add_technical_indicators` function to ensure correct calculation and integration
of technical indicators.

Tests:
    - Ensures the DataFrame includes the expected indicator columns.
    - Validates that rows with insufficient data for indicators are dropped.
    - Optionally checks the accuracy of specific indicator calculations.

Usage:
    Run this script using pytest:
        pytest test_indicators.py
"""
import os
import pandas as pd
from src.indicators import add_technical_indicators

def test_add_technical_indicators():
    """
    Test the `add_technical_indicators` function.

    Validates:
        - The DataFrame includes expected indicator columns after processing.
        - The resulting DataFrame contains no NaN values (dropped rows are expected).
        - Specific indicator calculations (e.g., SMA_20) are accurate (optional validation).

    Steps:
        1. Create a sample dataset with hourly close prices.
        2. Apply the `add_technical_indicators` function to generate indicators.
        3. Verify the presence of expected columns and absence of NaN values.
        4. Optionally validate individual indicator values for correctness.

    Asserts:
        - All expected columns (e.g., SMA, RSI, MACD) exist in the resulting DataFrame.
        - No NaN values remain in the DataFrame.
        - SMA_20 calculations match the expected mean of the last 20 values.

    Example:
        pytest test_indicators.py
    """
    # Create a small sample dataset
    data = pd.DataFrame({
        "Close_1h": [100, 102, 101, 103, 105, 107, 109, 108, 110, 112, 115, 117, 119, 118, 120]
    })

    # Apply the indicators
    data_with_indicators = add_technical_indicators(data)

    # Check that the new columns exist
    expected_columns = ["sma_20", "sma_50", "rsi", "macd", "signal_line"]
    for col in expected_columns:
        assert col in data_with_indicators.columns, f"{col} is missing from the DataFrame."

    # Check that there are no NaN values (NaNs should be dropped in the script)
    assert not data_with_indicators.isnull().values.any(), "DataFrame contains NaN values."

    # Optional: Validate specific indicator calculations (e.g., sma_20)
    if len(data_with_indicators) >= 20:
        sma_20_calculated = data_with_indicators["sma_20"].iloc[-1]
        expected_sma_20 = data["Close_1h"].iloc[-20:].mean()
        assert abs(sma_20_calculated - expected_sma_20) < 1e-6, "SMA_20 is not calculated correctly."
