from datetime import datetime

# The start date for fetching the historical data
start_date = '2020-01-01'

# The end date for fetching the historical data
end_date = datetime.today().strftime('%Y-%m-%d')

# The column name in the fetched data that will be predicted
target_column_name = 'open_price'

# Ticker symbol for the stock to predict. This is used to fetch data from Yahoo Finance.
yfinance_symbol = "ES=F"

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

    # This parameter allows the model to automatically drop old data that exceeds this number of periods.
    # This can be useful for very large datasets or for datasets with missing periods.
    'drop_data_older_than_periods': 200,

    # The number of jobs to run in parallel. Options: -1 (using all processors), or any positive integer
    'n_jobs': -1,
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
