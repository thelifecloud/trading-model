"""
test_data_pipeline.py

This module contains unit tests for the data_pipeline module.

Tests:
    - test_load_csv_data: Verifies the load_csv_data function correctly loads hourly and daily CSV files.
    - test_merge_and_clean_data: Verifies the merge_and_clean_data function merges and cleans data as expected.

Usage:
    Run this script using pytest:
        pytest test_data_pipeline.py
"""
import pandas as pd
from src.data_pipeline import load_csv_data, merge_and_clean_data

# Define file paths for testing
HOURLY_FILE = "/Users/lifecloud/Desktop/tradingmodel/trading_model/data/Binance_BTCUSDT_1h.csv"
DAILY_FILE = "/Users/lifecloud/Desktop/tradingmodel/trading_model/data/Binance_BTCUSDT_d.csv"

def test_load_csv_data():
    """
    Test the load_csv_data function.

    Validates:
        - Hourly and daily CSV files are loaded as pandas DataFrames.
        - Required columns are present in the loaded DataFrames.
    
    Asserts:
        - Both hourly and daily data are instances of pd.DataFrame.
        - All expected columns are present in both DataFrames.
    """
    hourly_data, daily_data = load_csv_data(HOURLY_FILE, DAILY_FILE)

    # Assert that data is loaded correctly
    assert isinstance(hourly_data, pd.DataFrame), "Hourly data should be a DataFrame."
    assert isinstance(daily_data, pd.DataFrame), "Daily data should be a DataFrame."

    # Check columns
    expected_columns = ["Unix", "Date", "Symbol", "Open", "High", "Low", "Close", "Volume BTC", "Volume USDT", "tradecount"]
    assert all(col in hourly_data.columns for col in expected_columns), "Hourly data is missing expected columns."
    assert all(col in daily_data.columns for col in expected_columns), "Daily data is missing expected columns."

def test_merge_and_clean_data():
    """
    Test the merge_and_clean_data function.

    Validates:
        - Hourly and daily data are merged correctly.
        - Output DataFrame contains the expected columns.
        - Output DataFrame is not empty after merging and cleaning.

    Asserts:
        - Combined data is an instance of pd.DataFrame.
        - All expected columns are present in the merged DataFrame.
        - Combined data is not empty after merging and cleaning.
    """
    # Load the data again for this test
    hourly_data, daily_data = load_csv_data(HOURLY_FILE, DAILY_FILE)
    combined_data = merge_and_clean_data(hourly_data, daily_data)

    # Assert that combined data is a DataFrame
    assert isinstance(combined_data, pd.DataFrame), "Combined data should be a DataFrame."

    # Assert expected columns in the merged output
    expected_columns = ["Unix_1h", "Open_1h", "High_1h", "Low_1h", "Close_1h", "Volume BTC_1h", 
                        "Volume USDT_1h", "tradecount_1h", "Unix_d", "Open_d", "High_d", 
                        "Low_d", "Close_d", "Volume BTC_d", "Volume USDT_d", "tradecount_d"]
    assert all(col in combined_data.columns for col in expected_columns), "Combined data is missing expected columns."

    # Ensure data is not empty after merging
    assert not combined_data.empty, "Combined data should not be empty."
