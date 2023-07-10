# config.py
from datetime import datetime
# Number of previous time steps to use as input features
forecast_steps = 21

target_column_name = 'Open'

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

# AutoML settings
automl_settings = {
#   'max_runtime_secs': 3600,              # Maximum time for AutoML to run (in seconds)
    'max_models': 20,                    # Maximum number of models to build
    'seed': 1,                             # Random seed for reproducibility
    'balance_classes': False,
    'include_algos': ['DRF', 'GBM', 'XGBoost', 'DeepLearning'],  # Algorithms to include in AutoML
    'keep_cross_validation_models': True,   # Whether to keep cross-validated models in the AutoML leaderboard
    'keep_cross_validation_predictions': True,  # Whether to keep cross-validated predictions in the AutoML leaderboard
    'verbosity': 'info'                     # Set the verbosity level of the AutoML process
}
start_date = '2019-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
# Other configuration options...
