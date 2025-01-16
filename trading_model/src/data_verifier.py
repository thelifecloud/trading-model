"""
data_verifier.py

This module provides utilities to verify and summarize data integrity in pandas DataFrames.

Key Features:
    - Check for missing values.
    - Validate column data types against expected types.
    - Provide a quick summary of dataset structure and content.
    - Run comprehensive data verification checks for preprocessing.

Functions:
    - check_missing_values: Identifies missing values in a DataFrame.
    - check_column_types: Verifies column data types match expectations.
    - print_dataset_summary: Prints a quick overview of the dataset for inspection.
    - verify_dataset: Runs a set of checks to ensure dataset integrity.
"""
def check_missing_values(df):
    """
    Check for missing values in the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Summary of missing values, including:
            - total_missing (int): Total number of missing values in the dataset.
            - missing_by_column (dict): Mapping of column names to counts of missing values.
    """
    missing_values = df.isna().sum()
    total_missing = missing_values.sum()
    missing_summary = {
        "total_missing": total_missing,
        "missing_by_column": missing_values[missing_values > 0].to_dict(),
    }
    return missing_summary

def check_column_types(df, expected_types):
    """
    Verify that columns match expected data types.

    Args:
        df (pd.DataFrame): Input DataFrame.
        expected_types (dict): Dictionary of expected column types.
            Example: {"price": "float64", "date": "datetime64"}

    Returns:
        list: Columns that do not match expected types, including:
            - column name,
            - actual type (or None if column is missing),
            - expected type.
    """
    mismatched = []
    for column, expected_type in expected_types.items():
        if column in df.columns:
            actual_type = str(df[column].dtype)
            if actual_type != expected_type:
                mismatched.append((column, actual_type, expected_type))
        else:
            mismatched.append((column, None, expected_type))
    return mismatched

def print_dataset_summary(df):
    """
    Print a summary of the dataset for quick inspection.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Prints:
        - Dataset shape (rows, columns).
        - Missing values summary.
        - Column data types.
    """
    print("\nDataset Summary:")
    print(f"Shape: {df.shape}")
    print("\nMissing Values:")
    print(check_missing_values(df))
    print("\nData Types:")
    print(df.dtypes)

def verify_dataset(df, expected_types=None):
    """
    Run all dataset verification checks.

    Args:
        df (pd.DataFrame): Input DataFrame.
        expected_types (dict, optional): Expected column types. Defaults to None.

    Returns:
        dict: Results of the verification checks, including:
            - missing_values: Summary of missing values.
            - type_mismatches (optional): List of columns with mismatched data types.
    """
    results = {
        "missing_values": check_missing_values(df),
    }
    if expected_types:
        results["type_mismatches"] = check_column_types(df, expected_types)
    return results