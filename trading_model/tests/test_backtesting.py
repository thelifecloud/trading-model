"""
test_backtesting.py

This module contains unit tests for the `backtesting` module, specifically testing the
`simulate_trading` function to ensure accurate simulation of trading strategies.

Tests:
    - test_simulate_trading: Validates the `simulate_trading` function with a simple mock model and
      a sample dataset.

Usage:
    Run this script using pytest:
        pytest test_backtesting.py
"""
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.backtesting import simulate_trading

def test_simulate_trading():
    """
    Test the `simulate_trading` function.

    Validates:
        - The function calculates trading performance metrics correctly.
        - Simulated trading results include expected profit and loss values.

    Steps:
        1. Create a small sample dataset with mock trading data.
        2. Define a simple mock model to generate predictions based on a predefined rule.
        3. Run the `simulate_trading` function with the dataset and mock model.
        4. Verify the calculated metrics and trading log.

    Asserts:
        - Metrics include total profit, win rate, Sharpe ratio, and final balance.
        - The trading results align with the expected outcomes based on the mock model.

    Example:
        pytest test_backtesting.py
    """
    # Create a small sample dataset
    data = pd.DataFrame({
        "Close_1h": [100, 102, 101, 105, 110],
        "sma_20": [95, 96, 97, 98, 99],
        "sma_50": [90, 91, 92, 93, 94],
        "macd": [1, 2, 3, 4, 5],
        "signal_line": [0, 1, 2, 3, 4],
        "target": [0, 1, 0, 1, 0],
    })

    # Simulate a simple model
    class SimpleModel:
        def predict(self, X):
            # Predict based on a simple rule
            return [1 if x > 100 else 0 for x in X["Close_1h"]]
