# main.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
import yfinance as yf
import psycopg2
import numpy as np
import pandas as pd
import logging
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import StandardScaler
import pandas_datareader as pdr
from pandas.tseries.holiday import USFederalHolidayCalendar
import data_fetching
import db_operations
import model_evaluation
from model_training import train_model  # Import the train_model function
import config
import h2o
from h2o.automl import H2OAutoML

forecast_steps = config.forecast_steps
# Initialize the H2O cluster
h2o.init()

# Set up logging
logging.basicConfig(filename='next1.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set the format for console output
console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console_handler.setFormatter(console_formatter)

# Add the console handler to the logger
logging.getLogger('').addHandler(console_handler)

logging.info('Starting script')

try:
    # Fetch and preprocess data
    logging.info('Fetching and preprocessing data')
    X_train, Y_train, X_test, Y_test, train_features, test_features, data, scaled_train_target, scaled_test_target, forecast_steps, target_scaler, num_features = data_fetching.fetch_and_preprocess_data()

    # Check that the shapes of the input data are as expected
    assert X_train.shape[1] == forecast_steps, 'Unexpected shape of X_train'
    assert X_test.shape[1] == forecast_steps, 'Unexpected shape of X_test'
    assert Y_train.ndim == 1, 'Unexpected shape of Y_train'
    assert Y_test.ndim == 1, 'Unexpected shape of Y_test'

    # Reshape the data to 2D
    X_train_2d = X_train.reshape((X_train.shape[0], -1))
    X_test_2d = X_test.reshape((X_test.shape[0], -1))

    logging.debug("Shape of X_train: ", X_train.shape)
    logging.debug("First few items of X_train: ", X_train[:5])

    # Convert numpy arrays to H2O Frame
    X_train_h2o = h2o.H2OFrame(X_train_2d.tolist())
    Y_train_h2o = h2o.H2OFrame(Y_train)
    X_test_h2o = h2o.H2OFrame(X_test_2d.tolist())
    Y_test_h2o = h2o.H2OFrame(Y_test)

    # Combine features and target into a single data frame
    train_data = X_train_h2o.cbind(Y_train_h2o)
    test_data = X_test_h2o.cbind(Y_test_h2o)

    # Print the column names
    logging.debug(train_data.columns)

    # Define the column names
    x = train_data.columns
    y = config.target_column_name
    x.remove(y)

    # Run AutoML
    aml = H2OAutoML(max_models=20, seed=1)
    aml.train(x=x, y=y, training_frame=train_data)

    # Get the best model
    model = aml.leader

    # Make predictions
    train_preds = model.predict(train_data)
    test_preds = model.predict(test_data)

    # Convert predictions to numpy array and flatten them
    train_preds = train_preds.as_data_frame().values.flatten()
    test_preds = test_preds.as_data_frame().values.flatten()

    # Log the shapes for debugging
    logging.debug(f'Shape of train_preds: {train_preds.shape}')
    logging.debug(f'Shape of Y_train: {Y_train.shape}')
    logging.debug(f'Shape of test_preds: {test_preds.shape}')
    logging.debug(f'Shape of Y_test: {Y_test.shape}')

    # Check that the shapes of the predictions are as expected
    assert train_preds.shape == Y_train.shape, 'Unexpected shape of train_preds'
    assert test_preds.shape == Y_test.shape, 'Unexpected shape of test_preds'

    # Print or log the predictions
    logging.info(f'Train predictions: {train_preds}')
    logging.info(f'Test predictions: {test_preds}')

    # Evaluate model
    train_predict, test_predict, predicted_test, train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2 = model_evaluation.evaluate_model(Y_train, Y_test, train_preds, test_preds)

    # Connect to the database
    conn, cur = db_operations.connect_to_db()

    # Create tables
    db_operations.create_tables(cur)

    # Generate new input data for forecast
    forecast_input = X_test[-forecast_steps:]  # Get the most recent observations

    # Convert forecast input to H2O data frame
    forecast_input_h2o = h2o.H2OFrame(forecast_input.tolist())

    # Make forecast
    forecast = model.predict(forecast_input_h2o)

    # Convert forecast to numpy array
    forecast = forecast.as_data_frame().values

    # Insert data
    db_operations.insert_data(cur, Y_train, train_predict, test_predict, forecast, target_scaler)

    # Insert forecast into the database
    db_operations.insert_forecast(cur, forecast)

    # Insert evaluation results
    db_operations.insert_evaluation_results(cur, train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2)

    # Commit changes
    conn.commit()

    # Close connection
    db_operations.close_connection(conn)

except Exception as e:
    logging.error(f'An error occurred: {e}')
    # Optionally, you can raise the exception again to stop the script
    raise

logging.info('Script completed')
