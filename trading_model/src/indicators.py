"""
indicators.py

This module provides functionality for adding technical indicators to a DataFrame.
These indicators are commonly used in trading models to analyze price movements and
identify potential buy/sell signals.

Key Features:
    - Simple Moving Averages (SMA) for trend analysis.
    - Relative Strength Index (RSI) for momentum evaluation.
    - Moving Average Convergence Divergence (MACD) and Signal Line for trend reversal detection.

Use Case:
    - Enhance raw price data with meaningful technical indicators for trading strategy development.
"""

import pandas as pd

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the provided DataFrame.

    Indicators Added:
        - Simple Moving Averages (SMA): 20-period and 50-period.
        - Relative Strength Index (RSI): 14-period.
        - Moving Average Convergence Divergence (MACD): Fast EMA (12), Slow EMA (26), and Signal Line (9).

    Parameters:
        data (pd.DataFrame): Input DataFrame containing a 'Close_1h' column for hourly close prices.

    Returns:
        pd.DataFrame: The input DataFrame with added technical indicators.

    Notes:
        - The function modifies the input DataFrame in place.
        - Rows with insufficient data to calculate indicators are dropped.

    Example:
        data = pd.DataFrame({"Close_1h": [100, 101, 102, 103, ...]})
        data_with_indicators = add_technical_indicators(data)
    """
    # Simple Moving Averages (SMA)
    data["sma_20"] = data["Close_1h"].rolling(window=20).mean()
    data["sma_50"] = data["Close_1h"].rolling(window=50).mean()

    # Relative Strength Index (RSI)
    close_diff = data["Close_1h"].diff(1)
    gain = close_diff.clip(lower=0).rolling(14).mean()
    loss = -close_diff.clip(upper=0).rolling(14).mean()
    data["rsi"] = 100 - (100 / (1 + (gain / loss)))

    # Moving Average Convergence Divergence (MACD)
    ema_12 = data["Close_1h"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close_1h"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema_12 - ema_26
    data["signal_line"] = data["macd"].ewm(span=9, adjust=False).mean()
    
    # Drop rows with NaNs
    data.dropna(inplace=True)
    
    return data
