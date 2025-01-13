import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.data_pipeline import load_csv_data, merge_and_clean_data


# Test files
hourly_file = "/Users/lifecloud/Desktop/tradingmodel/trading_model/data/Binance_BTCUSDT_1h.csv"
daily_file = "/Users/lifecloud/Desktop/tradingmodel/trading_model/data/Binance_BTCUSDT_d.csv"

# Test the functions
hourly_data, daily_data = load_csv_data(hourly_file, daily_file)
if hourly_data is not None and daily_data is not None:
    combined_data = merge_and_clean_data(hourly_data, daily_data)
    print("Combined DataFrame Head:")
    print(combined_data.head())
else:
    print("One or both files are missing.")
