# main.py
import logging
import pandas as pd
import config
import numpy as np
from model_training import train_model
import db_operations
from fetch_data import fetch_data

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

# Log to file as well
logging.info('Starting script')

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

    # Insert fetched data into the database
    db_operations.insert_fetched_data(cur, execution_id, data)

    # Commit changes
    conn.commit()

    # Assign the fetched data to the features variable
    features = data

    # Call the train_model function and get the results
    model, prediction, X_train_pd, X_test_pd, Y_train_pd, Y_test_pd, evaluation = train_model(data, config.autots_params['forecast_length'])
    # Log shapes for debugging
    logging.info('Shape of data: %s', np.shape(data))
    logging.info('Shape of prediction: %s', np.shape(prediction))

    # Connect to the database
    conn, cur = db_operations.connect_to_db()
    
    # Insert data
    db_operations.insert_data(cur, execution_id, data, prediction)
    
    # Insert the evaluation results into the database
    logging.info(f"Evaluation results:\n{evaluation}")
    db_operations.insert_evaluation_results(cur, execution_id, evaluation)
    
    # Commit changes
    conn.commit()

except ValueError as ve:
    logging.error('ValueError occurred: %s', ve)
except IOError as ioe:
    logging.error('IOError occurred: %s', ioe)
except Exception as e:
    logging.error('An error occurred: %s', e)
    # Optionally, you can raise the exception again to stop the script
    raise
finally:
    # Close connection
    db_operations.close_connection(conn)

logging.info('Script completed')