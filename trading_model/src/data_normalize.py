"""
data_normalize.py

This module provides functionality for normalizing numerical data in pandas DataFrames.

Features:
    - Normalize numerical data using Min-Max scaling or Z-score normalization.
    - Supports selective normalization for specific columns or all numerical columns.

Use Cases:
    - Preprocessing data for machine learning models.
    - Standardizing data to a common scale for better interpretability and model performance.

Notes:
    - If Min-Max scaling unsuitable to interpret original magnitudes consider Z-score normalization.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data(data: pd.DataFrame, method: str = "minmax", columns: list=None) -> pd.DataFrame:
    """
    Normalize numerical data using the specified method.

    Parameters:
        data (pd.DataFrame): Input DataFrame with numerical columns to normalize.
        method (str): Normalization method. Options are:
            - "minmax": Scales values to the range [0, 1].
            - "zscore": Standardizes values to have a mean of 0 and standard deviation of 1.
        columns (list, optional): List of columns to normalize. If None, all numerical columns are normalized.

    Returns:
        pd.DataFrame: DataFrame with normalized data.

    Raises:
        ValueError: If an unsupported normalization method is specified.

    Examples:
        # Normalize all numerical columns using Min-Max scaling
        normalized_data = normalize_data(data, method="minmax")
        
        # Normalize specific columns using Z-score normalization
        normalized_data = normalize_data(data, method="zscore", columns=["price", "volume"])
    """
    if columns is None:
        # Select all numerical columns
        columns = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "zscore":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    # Apply the scaler to the specified columns
    data[columns] = scaler.fit_transform(data[columns])
    
    return data
