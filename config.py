from datetime import datetime
import logging

# Log level for the root logger
log_level = logging.INFO
# The start date for fetching the historical data
start_date = '2021-01-01'

# The end date for fetching the historical data
end_date = datetime.today().strftime('%Y-%m-%d')

# The column name in the fetched data that will be predicted
target_column_name = 'open_price'

# Ticker symbol for the stock to predict. This is used to fetch data from Yahoo Finance.
yfinance_symbol = "ES=F"

# Features to be used for modeling
features_to_include = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA', 'EMA', 'RSI', 'MACD', 'Upper', 'Lower', 'Cumulative_Returns', 'VWAP', 'GDP Change', 'Unemployment Change', 'Target']

# The filename of the saved model template
template_filename = 'autots_template.csv'

# Set initial_training to True if you are training for the first time
# or if you want to start training from scratch. Set it to False if
# you want to load an existing model and continue training or make predictions.
initial_training = False

# Set evolve to True if you want the model to evolve over time, i.e., continue
# training from where it left off. Set it to False if you want to keep the model fixed.
evolve = True

# Parameters for the AutoTS model
autots_params = {
    # Number of periods into the future to forecast
    'forecast_length': 21,

    # Frequency of the data. Options: 'B' (business day frequency), 'D' (daily frequency), 'W' (weekly frequency), 'M' (monthly frequency)
    'frequency': 'B',

    # Method to ensemble models. Options: 'simple' (equally-weighted model average)
    'ensemble': 'simple',

    # Maximum number of generations to run. Each generation is a complete run through all selected models.
    'max_generations': 1,

    # The number of jobs to run in parallel. Options: -1 (using all processors), or any positive integer
    'n_jobs': -1,
}

# Parameters for creating the future regressor
create_regressor_params = {
    # Number of periods into the future to forecast
    'forecast_length': autots_params['forecast_length'],
    
    # Frequency of the data
    'frequency': autots_params['frequency'],
    
    # Number of most recent periods to drop from the future regressor
    'drop_most_recent': 0,
    
    # Whether to scale the future regressor
    'scale': True,
    
    # Method to summarize the future regressor
    'summarize': 'auto',
    
    # Method to backfill missing data in the future regressor
    'backfill': 'bfill',
    
    # Method to fill NA values in the future regressor
    'fill_na': 'spline',
    
    # Countries to consider for holidays in the future regressor
    'holiday_countries': {'US': None},  # requires holidays package
    
    # Whether to encode the type of holiday in the future regressor
    'encode_holiday_type': True,
    
    # Method to encode date parts in the future regressor
    'datepart_method': 'simple_2',
}

# Details for connecting to the database
database = {
    # Host name of the database
    "host": "snuffleupagus.db.elephantsql.com",
    
    # Name of the database
    "database": "rzpjtxcf",
    
    # Username for the database
    "user": "rzpjtxcf",
    
    # Password for the database
    "password": "lbFXUWGzaOw_aju7fmq0mNkt39T3fAKf"
}
