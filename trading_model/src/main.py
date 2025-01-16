"""
main.py

This script serves as a manual test driver for the trading model pipeline. It sequentially 
executes the following steps to test and demonstrate the functionality of the system:

Steps:
    1. Load and merge hourly and daily data from CSV files.
    2. Add technical indicators to the dataset.
    3. Normalize the data using Min-Max scaling or Z-score normalization.
    4. Add a target column based on defined indicator confluence logic.
    5. Prepare features and target variables for machine learning.
    6. Split the data into training and testing sets.
    7. Train a Random Forest Classifier on the training data.
    8. Evaluate the trained model on the test data.
    9. Backtest the trading strategy using model predictions.
    10. Save processed data and visualize results.

Use Case:
    - Provides a single end-to-end script to verify the functionality of all modules in the 
      trading model pipeline.
    - Demonstrates how the modules can be integrated to build, test, and evaluate a trading model.

Usage:
    Run the script directly:
        python main.py
"""
from sklearn.metrics import confusion_matrix
from src.data_pipeline import load_csv_data, merge_and_clean_data
from src.indicators import add_technical_indicators
from src.data_normalize import normalize_data
from src.target_creation import add_target
from src.models import prepare_features_and_target, split_data, train_model, evaluate_model
from src.backtesting import simulate_trading
from src.visualization import plot_feature_importance, plot_trading_performance, plot_confusion_matrix


def main():
    # File paths for your data
    hourly_file = "/Users/lifecloud/Desktop/tradingmodel/trading_model/data/Binance_BTCUSDT_1h.csv"
    daily_file = "/Users/lifecloud/Desktop/tradingmodel/trading_model/data/Binance_BTCUSDT_d.csv"

    # Step 1: Load and merge data
    print("Loading and merging data...")
    hourly_data, daily_data = load_csv_data(hourly_file, daily_file)
    merged_data = merge_and_clean_data(hourly_data, daily_data)
    print("Data merged successfully.")

    # Step 2: Add indicators
    print("Adding technical indicators...")
    merged_data = add_technical_indicators(merged_data)
    print("Indicators added successfully.")

    # Step 3: Normalize data
    print("Normalizing data...")
    merged_data = normalize_data(merged_data, method="minmax")  # Change to "zscore" if needed
    print("Data normalized successfully.")

    # Step 3.5: Add target
    print("Adding target column...")
    merged_data = add_target(merged_data)
    print("Target column added.")

    # Step 4: Prepare features and target
    print("Preparing features and target...")
    X, y = prepare_features_and_target(merged_data, target_column="target")
    print("Features and target prepared.")

    # Step 5: Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Testing set: {X_test.shape}, {y_test.shape}")

    # Step 6: Train the model
    print("Training the model...")
    model = train_model(X_train, y_train)
    print("Model trained successfully.")

    # Step 7: Evaluate the model
    print("Evaluating the model...")
    metrics = evaluate_model(model, X_test, y_test)
    print("Model Evaluation:")
    print("Accuracy:", metrics["accuracy"])
    print("Classification Report:\n", metrics["classification_report"])
    print("Confusion Matrix:\n", metrics["confusion_matrix"])   

    # Step 8: Backtest the model
    print("Backtesting the model...")
    metrics, backtest_results = simulate_trading(merged_data, model, initial_capital=10000, profit_target=0.02, stop_loss=0.01)
    print("Backtest Metrics:")
    print("Total Profit:", metrics["total_profit"])
    print("Win Rate:", metrics["win_rate"])
    print("Sharpe Ratio:", metrics["sharpe_ratio"])
    print("Final Balance:", metrics["final_balance"])

    # Step 9: Save or inspect results
    output_file = "/Users/lifecloud/Desktop/tradingmodel/trading_model/data/processed_data.csv"
    merged_data.to_csv(output_file)
    print(f"Processed data saved to {output_file}")

    # Step 10: Manual visualization test
    print("Running visualization tests...")
    if hasattr(model, "feature_importances_"):
        print("Plotting feature importance...")
        plot_feature_importance(model.feature_importances_, X_train.columns)

    print("Plotting trading performance...")
    plot_trading_performance(backtest_results)

    print("Plotting confusion matrix...")
    y_pred = model.predict(X_test.select_dtypes(include=["number"]))
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_labels=["No Buy", "Buy"])

if __name__ == "__main__":
    main()
