"""
backtesting.py

This module functions to simulate trading strategies using machine learning model predictions. 
It includes entry and exit logic to calculate trading performance metrics.

Key Features:
    - Simulate trades based on model predictions.
    - Compute metrics such as total profit, win rate, and Sharpe ratio.
    - Generate a detailed trade log and profit/loss (PnL) for each step.

Use Case:
    - Backtest trading strategies to evaluate performance and refine trading models.
"""

import numpy as np

def simulate_trading(data, model, initial_capital=10000, risk_per_trade=0.01, profit_target=0.02, stop_loss=0.01):
    """
    Simulate a trading strategy based on model predictions with entry and exit logic.

    Parameters:
        data (pd.DataFrame): Dataset with features, target, and predictions.
            - Must include a "Close_1h" column for trade price.
        model: Trained machine learning model for generating predictions.
        initial_capital (float): Starting capital for the strategy. Default is 10000.
        risk_per_trade (float): Percentage of capital risked per trade. Default is 0.01 (1%).
        profit_target (float): Profit target as a percentage. Default is 0.02 (2%).
        stop_loss (float): Stop loss as a percentage. Default is 0.01 (1%).

    Returns:
        tuple:
            - dict: Trading performance metrics, including:
                - total_profit (float): Total profit or loss from trading.
                - win_rate (float): Percentage of profitable trades.
                - sharpe_ratio (float): Risk-adjusted return metric.
                - final_balance (float): Ending capital after simulation.
            - pd.DataFrame: DataFrame with trading results, including:
                - predicted (int): Model predictions (1 for buy, 0 for no action).
                - PnL (float): Profit or loss for each trade.

    Notes:
        - The function assumes a "Close_1h" column is present in the data for trade pricing.
        - Predictions are generated if the "predicted" column is missing.

    Example:
        metrics, result_data = simulate_trading(data, trained_model)
        print("Total Profit:", metrics["total_profit"])
        print("Final Balance:", metrics["final_balance"])
    """
    # Ensure the time column is included
    data = data.copy()
    if "time" not in data.columns:
        data["time"] = range(len(data))  # Add a dummy time column if missing

    # Ensure data contains features and target
    X = data.drop(columns=["time", "target", "predicted", "PnL"], errors="ignore")


    # Generate predictions if not already done
    if "predicted" not in data.columns:
        data["predicted"] = model.predict(X)

    # Initialize trading metrics
    balance = initial_capital
    position = 0  # Units held
    entry_price = 0
    trade_log = []  # To track individual trades
    data["PnL"] = 0.0

    # Simulate trades
    for i in range(len(data)):
        price = data["Close_1h"].iloc[i]
        prediction = data["predicted"].iloc[i]

        # Entry logic
        if prediction == 1 and position == 0:  # Enter a trade
            position = balance * risk_per_trade / price  # Number of units bought
            entry_price = price
            balance -= position * price  # Deduct cost from balance

        # Exit logic
        if position > 0:
            # Check for profit target or stop loss
            if (price / entry_price - 1) >= profit_target or (price / entry_price - 1) <= -stop_loss:
                profit = position * (price - entry_price)
                balance += profit  # Add profit/loss to balance
                data.loc[i, "PnL"] = profit
                trade_log.append(profit)
                position = 0  # Reset position

    # Final metrics
    total_profit = balance - initial_capital
    win_rate = np.mean([1 if p > 0 else 0 for p in trade_log]) if trade_log else 0
    sharpe_ratio = np.mean(trade_log) / np.std(trade_log) if len(trade_log) > 1 else 0

    metrics = {
        "total_profit": total_profit,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "final_balance": balance,
    }

    return metrics, data
