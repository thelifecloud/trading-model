"""
models.py

This module provides functionality for building, training, and evaluating machine learning models,
specifically using the Random Forest Classifier for trading data.

Key Features:
    - Prepares features and target variables for modeling.
    - Splits data into training and testing sets.
    - Trains a Random Forest Classifier on numerical features.
    - Evaluates the model's performance using accuracy, classification reports, and confusion matrices.

Use Case:
    - Develop and validate predictive models for trading strategies.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def prepare_features_and_target(data, target_column):
    """
    Prepare features (X) and target (y) for modeling.

    Parameters:
        data (pd.DataFrame): Dataset containing features and the target column.
        target_column (str): Name of the target column.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): Feature set with all columns except the target.
            - y (pd.Series): Target variable column.

    Example:
        data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        X, y = prepare_features_and_target(data, "target")
    """
    features = data.drop(columns=[target_column]).columns
    X = data[features]
    y = data[target_column]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        test_size (float, optional): Proportion of data to use as test set. Default is 0.2.
        random_state (int, optional): Random state for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training feature set.
            - X_test (pd.DataFrame): Testing feature set.
            - y_train (pd.Series): Training target variable.
            - y_test (pd.Series): Testing target variable.

    Example:
        X_train, X_test, y_train, y_test = split_data(X, y)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, random_state=42):
    """
    Train a Random Forest Classifier on the training data.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target variable.
        random_state (int, optional): Random state for reproducibility. Default is 42.

    Returns:
        RandomForestClassifier: A trained RandomForestClassifier model.

    Notes:
        - Only numeric features in `X_train` are used for training.

    Example:
        model = train_model(X_train, y_train)
    """
    X_train_numeric = X_train.select_dtypes(include=["number"])  # Use numeric columns only
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train_numeric, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and return performance metrics.

    Parameters:
        model (RandomForestClassifier): Trained model.
        X_test (pd.DataFrame): Testing feature set.
        y_test (pd.Series): Testing target variable.

    Returns:
        dict: A dictionary containing:
            - accuracy (float): Accuracy score of the model.
            - classification_report (str): Detailed classification report.
            - confusion_matrix (ndarray): Confusion matrix as a 2D array.

    Notes:
        - Only numeric features in `X_test` are used for evaluation.

    Example:
        metrics = evaluate_model(model, X_test, y_test)
    """
    X_test_numeric = X_test.select_dtypes(include=["number"])  # Align with training features
    y_pred = model.predict(X_test_numeric)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics
