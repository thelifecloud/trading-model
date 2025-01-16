"""
test_visualization.py

This module contains unit tests for the `visualization` module, specifically testing
the following plotting functions:
    - plot_feature_importance
    - plot_trading_performance
    - plot_confusion_matrix

Tests:
    - Ensure each function runs without errors using sample data.

Usage:
    Run this script using pytest:
        pytest test_visualization.py
"""
import os
import pandas as pd
import numpy as np
from src.visualization import plot_feature_importance, plot_trading_performance, plot_confusion_matrix

def test_plot_feature_importance():
    """
    Test the `plot_feature_importance` function to ensure it runs without errors.

    Validates:
        - The function executes without exceptions using a sample feature importance dataset.

    Example:
        feature_importances = [0.3, 0.5, 0.2]
        feature_names = ["Feature A", "Feature B", "Feature C"]
        plot_feature_importance(feature_importances, feature_names)
    """
    feature_importances = [0.3, 0.5, 0.2]
    feature_names = ["Feature A", "Feature B", "Feature C"]

    try:
        plot_feature_importance(feature_importances, feature_names)
    except Exception as e:
        assert False, f"plot_feature_importance raised an exception: {e}"

def test_plot_trading_performance():
    """
    Test the `plot_trading_performance` function to ensure it runs without errors.

    Validates:
        - The function executes without exceptions using a sample trading performance dataset.

    Example:
        trading_data = pd.DataFrame({
            "time": pd.date_range(start="2023-01-01", periods=5, freq="h"),
            "PnL": [100, -50, 200, -100, 150]
        })
        plot_trading_performance(trading_data)
    """
    trading_data = pd.DataFrame({
        "time": pd.date_range(start="2023-01-01", periods=5, freq="h"),
        "PnL": [100, -50, 200, -100, 150]
    })

    try:
        plot_trading_performance(trading_data)
    except Exception as e:
        assert False, f"plot_trading_performance raised an exception: {e}"

def test_plot_confusion_matrix():
    """
    Test the `plot_confusion_matrix` function to ensure it runs without errors.

    Validates:
        - The function executes without exceptions using a sample confusion matrix.

    Example:
        cm = np.array([[10, 2], [3, 5]])
        class_labels = ["No Buy", "Buy"]
        plot_confusion_matrix(cm, class_labels)
    """
    cm = np.array([[10, 2], [3, 5]])
    class_labels = ["No Buy", "Buy"]

    try:
        plot_confusion_matrix(cm, class_labels)
    except Exception as e:
        assert False, f"plot_confusion_matrix raised an exception: {e}"
