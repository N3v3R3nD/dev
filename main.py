# main.py
import logging
import numpy as np
from autots import create_regressor
from model_training import train_model
import db_operations
from fetch_data import fetch_data
import config

# Create a root logger
logger = logging.getLogger()
logger.setLevel(config.log_level)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set the format for console output
console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console_handler.setFormatter(console_formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Create a file handler
file_handler = logging.FileHandler('next1.log')
file_handler.setLevel(logging.DEBUG)

# Set the format for file output
file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(file_formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Now, logging.info(), logging.debug(), etc. should log to both the console and the file
logging.info('Starting script')

conn = None

try:
    # Connect to the database
    conn, cur = db_operations.connect_to_db()

    # Get the next execution_id
    execution_id = db_operations.get_next_execution_id(cur)

    # Create tables
    db_operations.create_tables(cur)

    # Fetch and preprocess data
    logging.info('Fetching data')
    data = fetch_data()

    # Create a future regressor
    regr_train, regr_fcst = create_regressor(
        data,
        **config.create_regressor_params
        )
    # Insert fetched data into the database
    db_operations.insert_fetched_data(cur, execution_id, data)

    # Commit changes
    conn.commit()

    # Assign the fetched data to the features variable
    features = data

    # Call the train_model function and get the results
    model, prediction, X_train_pd, X_test_pd, Y_train_pd, Y_test_pd, evaluation, prediction_df = train_model(
        data, config.autots_params['forecast_length'], regr_train
    )
    # Log shapes for debugging
    logging.debug('Shape of data: %s', np.shape(data))
    logging.debug('Shape of prediction: %s', np.shape(prediction_df))

    # Connect to the database
    conn, cur = db_operations.connect_to_db()

    # Insert forecast into the database
    db_operations.insert_forecast(cur, execution_id, prediction)

    # Insert Data into the database
    db_operations.insert_data(cur, execution_id, data, prediction_df)

    # Insert the evaluation results into the database
    db_operations.insert_evaluation_results(cur, execution_id, evaluation)

    # Commit changes
    conn.commit()

except ValueError as ve:
    logging.error('ValueError occurred: %s', ve)
except IOError as ioe:
    logging.error('IOError occurred: %s', ioe)
except Exception as error:
    logging.error('An error occurred: %s', error)
    # Optionally, you can raise the exception again to stop the script
    raise
finally:
    # Close connection if it was opened
    if conn is not None:
        db_operations.close_connection(conn)

logging.info('Script completed')