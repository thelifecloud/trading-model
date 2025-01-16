### Trading Model

This repository contains a modular, tested, and fully functional predictive trading model for cryptocurrency markets. The project was developed to automate trading strategies by combining machine learning and backtesting methods with real-world trading logic. Below is an overview of the project structure, functionality, and usage.

#### Features
1. Data Pipeline:
    - Cleans and preprocesses cryptocurrency market data (OHLCV format).
    - Merges hourly and daily datasets for a comprehensive view.

2. Indicators:
    - Calculates technical indicators such as moving averages and volume-based metrics.
    - Handles missing values and normalizes indicator values.

3. Modeling:
    - Trains a machine learning model to predict trading signals.
    - Includes feature engineering and target preparation logic.

4. Backtesting:
    - Simulates trading strategies based on model predictions.
    - Includes risk management, profit targets, and stop-loss logic.

5. Visualization:
    - Generates performance metrics and visualizations (confusion matrix, profit curves...).

6. Testing:
    - Fully tested with pytest to ensure functionality across all modules.

#### Project Structure
'''
trading_model/
├── data/                     # Contains input CSV files (e.g., Binance_BTCUSDT_1h.csv, Binance_BTCUSDT_d.csv).
├── notebooks/                # Jupyter notebooks for experimentation and analysis.
├── src/                      # Source code for the project.
│   ├── backtesting.py        # Simulates trading strategies based on predictions.
│   ├── data_pipeline.py      # Loads and preprocesses data.
│   ├── indicators.py         # Calculates technical indicators.
│   ├── main.py               # Test driver for manually testing modules.
│   ├── models.py             # Defines and trains the predictive model.
│   ├── plotting.py           # Visualization logic for metrics and results.
│   ├── sentiment_analysis.py # Placeholder for sentiment analysis (future feature).
├── tests/                    # Test scripts for each module.
│   ├── test_data_pipeline.py
│   ├── test_indicators.py
│   ├── test_models.py
│   ├── test_backtesting.py
│   ├── test_visualization.py
├── requirements.txt          # Python dependencies for the project.
├── README.md                 # Project overview (you are here).
'''

### Setup Instructions
#### Prerequisites

1. Python 3.8 or higher.
2. Git.
3. A virtual environment (recommended).

#### Installation

1. Clone the repository:
    git clone https://github.com/your-username/trading_model.git
    cd trading_model

2. Create and activate a virtual environment:
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
 
3. Install dependencies:
    pip install -r requirements.txt

### Usage
#### Running the Project
1. To manually test modules, use: 
    python -m src.main

2. To run tests
    pytest tests/

#### Input data requirements
1. Currently using explicit paths to CSV files on desktop
    - Ensure to specify your explicit path data_pipeline.py
    - Files sourced from https://www.cryptodatadownload.com/data/
    - If using your own CSV, required coloumns: Unix, Date, Symbol, Open, High, Low, Close, Volume BTC, Volume USDT, tradecount.

#### Features and Targets
1. Model uses technical indicators an dnormalized values as features
2. Targets are defined based on customizable trading logic

#### Visualization
1. After backtesting, metrics and performance plots can be generated using visualization.py
