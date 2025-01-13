# data pipeline for pulling from csv and cleaning
# we will make another data pipeline for pulling from API and cleaning too.
import pandas as pd

def load_csv_data(hourly_file, daily_file):
    """
    Load hourly and daily CSV files into pandas DataFrames.

    Parameters:
        hourly_file (str): Path to the hourly data CSV.
        daily_file (str): Path to the daily data CSV.

    Returns:
        pd.DataFrame, pd.DataFrame: Hourly and daily data DataFrames.
    """
    csv_files = [hourly_file, daily_file]
    dataframes = []

    for file in csv_files:
        try:
            df = pd.read_csv(file, skiprows=1)
            dataframes.append(df)
        except FileNotFoundError:
            print(f"Error: File not found - {file}")
            dataframes.append(None)

    return dataframes

def merge_and_clean_data(hourly_data, daily_data):
    """
    Merge hourly and daily data, and perform pre-indicator cleaning.

    Parameters:
        hourly_data (pd.DataFrame): DataFrame containing hourly data.
        daily_data (pd.DataFrame): DataFrame containing daily data.

    Returns:
        pd.DataFrame: Merged and cleaned DataFrame.
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
    merged_data.drop(columns=["Date_1h", "Symbol_1h", "Date_d", "Symbol_d"], inplace=True, errors="ignore")

    # Drop rows with NaN values
    merged_data.dropna(inplace=True)

    return merged_data

# Test usage:
if __name__ == "__main__":
    hourly_file = "Binance_BTCUSDT_1h.csv"
    daily_file = "Binance_BTCUSDT_d.csv"

    # Load data
    hourly_data, daily_data = load_csv_data(hourly_file, daily_file)

    # Merge and clean
    combined_data = merge_and_clean_data(hourly_data, daily_data)

    print(combined_data.head())
