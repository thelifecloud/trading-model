"""
target_creation.py

This module provides functionality for creating target columns in a dataset based on
indicator confluence. Targets are used in trading models to signal potential buy opportunities.

Key Features:
    - Add a binary target column to a DataFrame based on specified indicator conditions.

Use Case:
    - Create a labeled dataset for training a trading model by identifying periods
      where indicators align for a potential buy signal.
"""
import pandas as pd
def add_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary target column based on indicator confluence.

    Parameters:
        data (pd.DataFrame): DataFrame containing indicators such as SMA, MACD, and Signal Line.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'target' column where:
            - 1 indicates a buy signal (e.g., SMA(20) > SMA(50) and MACD > Signal Line).
            - 0 indicates no buy signal.

    Logic:
        - Default 'target' column value is 0.
        - Sets 'target' to 1 where:
            - SMA(20) is greater than SMA(50), and
            - MACD is greater than Signal Line.

    Example:
        data = pd.DataFrame({
            "sma_20": [1, 2, 3],
            "sma_50": [3, 2, 1],
            "macd": [0.5, 0.6, 0.4],
            "signal_line": [0.4, 0.5, 0.6]
        })
        updated_data = add_target(data)
        print(updated_data)
    """
    data["target"] = 0  # Default to 0 (No Buy)
    data.loc[(data["sma_20"] > data["sma_50"]) & (data["macd"] > data["signal_line"]), "target"] = 1
    return data
