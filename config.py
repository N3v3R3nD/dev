# config.py

# Number of previous time steps to use as input features
forecast_steps = 21

target_column_name = 'C1100'

# Ticker symbol for the stock to predict
yfinance_symbol = "SPY"

# Number of epochs with no improvement after which training will be stopped
early_stopping_patience = 5


# Details for connecting to the database
database = {
    # Host name
    "host": "localhost",
    # Database name
    "database": "stock",
    # Username
    "user": "postgres",
    # Password
    "password": "test123"
}

# Number of splits for time series cross-validation
tscv_splits = 10
