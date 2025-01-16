"""
visualization.py

This module provides functions to visualize key aspects of the trading model, including:
    - Feature importances from a trained model.
    - Trading performance over time.
    - Confusion matrix for classification results.

Key Features:
    - Plot and interpret feature importances for model understanding.
    - Visualize cumulative profit and loss (PnL) over time for strategy evaluation.
    - Display confusion matrix as a heatmap for classification performance analysis.

Use Case:
    - Use these visualizations to debug, interpret, and communicate the performance and behavior
      of trading models and strategies.
"""
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(feature_importances, feature_names, top_n=10):
    """
    Plot the top N most important features based on feature importances.

    Parameters:
        feature_importances (list or array): Importance values of the features.
        feature_names (list): Names of the features.
        top_n (int, optional): Number of top features to display. Default is 10.

    Example:
        plot_feature_importance([0.3, 0.2, 0.5], ["feature1", "feature2", "feature3"])
    """
    # Create a Series for better visualization
    importance_series = pd.Series(feature_importances, index=feature_names)
    top_features = importance_series.nlargest(top_n)

    plt.figure(figsize=(10, 6))
    top_features.plot(kind="barh", title="Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

def plot_trading_performance(trading_data):
    """
    Plot the performance of a trading strategy over time.

    Parameters:
        trading_data (pd.DataFrame): DataFrame containing columns:
            - 'time': Timestamps or indices of trading events.
            - 'PnL': Profit or loss for each trade.

    Notes:
        - Assumes the DataFrame includes a 'PnL' column and computes 'Cumulative PnL'.

    Example:
        plot_trading_performance(pd.DataFrame({
            "time": [1, 2, 3],
            "PnL": [100, -50, 200],
        }))
    """
    trading_data["Cumulative PnL"] = trading_data["PnL"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(trading_data["time"], trading_data["Cumulative PnL"], label="Cumulative PnL", alpha=0.8)
    plt.title("Trading Performance Over Time")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_labels):
    """
    Plot a confusion matrix as a heatmap.

    Parameters:
        cm (array): Confusion matrix values as a 2D array.
        class_labels (list): Labels for the classes.

    Example:
        cm = [[50, 10], [5, 35]]
        plot_confusion_matrix(cm, ["Class 0", "Class 1"])
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Annotate the cells
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
