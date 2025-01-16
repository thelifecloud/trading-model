"""
data_pipeline.py

This module handles data loading, merging, and cleaning for the trading model.

Key Features:
    - Loads hourly and daily data from CSV files.
    - Merges the data and performs pre-indicator cleaning steps.
    - Ensures compatibility and readiness for further processing by verifying required columns.

Functions:
    - load_csv_data: Loads hourly and daily data from specified CSV files into pandas DataFrames.
    - merge_and_clean_data: Merges hourly/daily data, cleans, and prepares it for further analysis.
"""
import os
import pandas as pd

def load_csv_data(hourly_file, daily_file):
    """
    Load hourly and daily CSV files into pandas DataFrames.

    Parameters:
        hourly_file (str): Path to the hourly data CSV file.
        daily_file (str): Path to the daily data CSV file.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - hourly_data (pd.DataFrame or None): DataFrame containing hourly data, or None if loading fails.
            - daily_data (pd.DataFrame or None): DataFrame containing daily data, or None if loading fails.

    Notes:
        - The function validates that required columns are present in each CSV file.
        - If a file cannot be loaded or has invalid columns, None is returned for that file, and an error is logged.
    """
    csv_files = [hourly_file, daily_file]
    dataframes = []

    for file in csv_files:
        try:
            df = pd.read_csv(file, skiprows=1)
            # Validate columns based on actual data
            required_cols={"Unix","Date","Symbol","Open","High","Low","Close","Volume BTC","Volume USDT","tradecount"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Missing required columns in {file}. Found: {df.columns}")
            dataframes.append(df)
        except Exception as e:
            print(f"Error in {file}: {e}")
            dataframes.append(None)

    # Explicitly unpack the list into a tuple
    hourly_data, daily_data = dataframes
    return hourly_data, daily_data

def merge_and_clean_data(hourly_data, daily_data):
    """
    Merge hourly and daily data, and perform pre-indicator cleaning.

    Parameters:
        hourly_data (pd.DataFrame): DataFrame containing hourly data.
        daily_data (pd.DataFrame): DataFrame containing daily data. 
        If None, only hourly data is returned.

    Returns:
        pd.DataFrame: A merged and cleaned DataFrame ready for further processing.

    Notes:
        - Converts the Unix timestamp to datetime for both hourly and daily data.
        - Merges data on the 'time' index, ensuring alignment across timeframes.
        - Drops unnecessary columns (e.g., duplicated symbols and dates).
        - Removes rows with NaN values to ensure data integrity.
        - If daily data is unavailable, proceeds with hourly data only.

    Raises:
        ValueError: If input data is invalid or merging fails.
    """
    if daily_data is None:
        print("Daily data is missing. Proceeding with hourly data only.")
        return hourly_data.copy()

    # Convert timestamps
    hourly_data["time"] = pd.to_datetime(hourly_data["Unix"], unit="ms")
    daily_data["time"] = pd.to_datetime(daily_data["Unix"], unit="ms")
    hourly_data.set_index("time", inplace=True)
    daily_data.set_index("time", inplace=True)

    # Merge hourly and daily data
    merged_data = hourly_data.join(daily_data, how="inner", lsuffix="_1h", rsuffix="_d")

    # Drop unnecessary columns
    merged_data.drop(columns=["Date_1h","Symbol_1h","Date_d","Symbol_d"],inplace=True,errors="ignore")

    # Drop rows with NaN values
    merged_data.dropna(inplace=True)

    return merged_data

# Standalone execution block for isolated manual testing and debugging
if __name__ == "__main__":
    # Absolute paths to your data folder
    hourly_file = "/Users/lifecloud/Desktop/tradingmodel/trading_model/data/Binance_BTCUSDT_1h.csv"
    daily_file = "/Users/lifecloud/Desktop/tradingmodel/trading_model/data/Binance_BTCUSDT_d.csv"

    print(f"Hourly file path: {hourly_file}")
    print(f"Daily file path: {daily_file}")
    print(f"Hourly file exists: {os.path.exists(hourly_file)}")
    print(f"Daily file exists: {os.path.exists(daily_file)}")


    # Load data
    hourly_data, daily_data = load_csv_data(hourly_file, daily_file)

    # Check if data is loaded
    if hourly_data is None:
        raise FileNotFoundError("Hourly data is missing. Cannot proceed without hourly data.")

    # Merge and clean
    try:
        combined_data = merge_and_clean_data(hourly_data, daily_data)
        print(combined_data.head())
    except ValueError as e:
        print(f"Error: {e}")