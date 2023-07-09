# main.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from sklearn.model_selection import KFold
from pandas.tseries.holiday import USFederalHolidayCalendar
import data_fetching
import db_operations
import model_evaluation
from model_training import train_model  # Import the train_model function
import config
import h2o
from h2o.automl import H2OAutoML

# Initialize the H2O cluster
h2o.init()

# Access the parameters
use_kfold = config.use_kfold
kfold_splits = config.kfold_splits
early_stopping_patience = config.early_stopping_patience
look_back = config.look_back
model_params = config.model_params

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

# Fetch and preprocess data
logging.info('Fetching and preprocessing data')
X_train, Y_train, X_test, Y_test, train_features, test_features, data, scaled_train_target, scaled_test_target, look_back, target_scaler, num_features = data_fetching.fetch_and_preprocess_data()

# Reshape the data to 2D
X_train_2d = X_train.reshape((X_train.shape[0], -1))
X_test_2d = X_test.reshape((X_test.shape[0], -1))

# Convert numpy arrays to H2O Frame
X_train_h2o = h2o.H2OFrame(X_train)
Y_train_h2o = h2o.H2OFrame(Y_train)
X_test_h2o = h2o.H2OFrame(X_test)
Y_test_h2o = h2o.H2OFrame(Y_test)


# Combine features and target into a single data frame
train_data = X_train_h2o.cbind(Y_train_h2o)
test_data = X_test_h2o.cbind(Y_test_h2o)

# Print the column names
print(train_data.columns)

# Define the column names
x = train_data.columns
y = "C1100"
x.remove(y)

# Run AutoML
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train_data)

# Get the best model
model = aml.leader

# Make predictions
preds = model.predict(test_data)

# Convert predictions to numpy array
preds = preds.as_data_frame().values

# Evaluate model
train_predict, test_predict, train_rmse, test_rmse, train_mae, test_mae = model_evaluation.evaluate_model(Y_train, Y_test, preds, preds, target_scaler)

# Connect to the database
conn, cur = db_operations.connect_to_db()

# Create tables
db_operations.create_tables(cur)

# Insert data
db_operations.insert_data(cur, history, Y_train, train_predict, test_predict, target_scaler)
db_operations.insert_evaluation_results(cur, train_rmse, test_rmse, train_mae, test_mae)

# Close the connection
db_operations.close_connection(conn)

logging.info('Script completed')
