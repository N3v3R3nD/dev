import psycopg2
from datetime import datetime, timedelta
import logging
import config
import numpy as np

# Extract database credentials from config
db_config = config.database
host = db_config['host']
database = db_config['database']
user = db_config['user']
password = db_config['password']

def connect_to_db():
    # Connect to the database
    logging.info('Connecting to the database')
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    cur = conn.cursor()
    return conn, cur

def create_tables(cur):
    # Create actual_vs_predicted table if it doesn't exist
    logging.info('Creating actual_vs_predicted table if it does not exist')
    cur.execute("""
        CREATE TABLE IF NOT EXISTS actual_vs_predicted (
            date DATE PRIMARY KEY,
            actual_price FLOAT,
            predicted_price FLOAT
        )
    """)

    # Create evaluation_results table if it doesn't exist
    logging.info('Creating evaluation_results table if it doesn\'t exist')
    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id SERIAL PRIMARY KEY,
            train_rmse FLOAT,
            test_rmse FLOAT,
            train_mae FLOAT,
            test_mae FLOAT,
            train_rae FLOAT,
            test_rae FLOAT,
            train_rse FLOAT,
            test_rse FLOAT,
            train_r2 FLOAT,
            test_r2 FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create forecasted_prices table if it doesn't exist
    logging.info('Creating forecasted_prices table if it doesn\'t exist')
    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecasted_prices (
            date DATE PRIMARY KEY,
            forecasted_price FLOAT
        )
    """)

def insert_data(cur, Y_train, train_predict, target_scaler):    
    # Insert actual and predicted prices into the database
    logging.info('Inserting actual and predicted prices into the database')

    if Y_train.shape[0] != train_predict.shape[0]:
        raise ValueError("Length of Y_train and train_predict don't match")

    for i in range(len(Y_train)):
        date = (datetime.today() - timedelta(days=len(Y_train) - i - 1)).strftime('%Y-%m-%d')  # Calculate the correct date
        actual_price = target_scaler.inverse_transform(Y_train[i].reshape(-1, 1))[0][0]
        predicted_price = train_predict[i]
        
        cur.execute(f"""
            INSERT INTO actual_vs_predicted (date, actual_price, predicted_price) 
            VALUES ('{date}', {actual_price}, {predicted_price}) 
            ON CONFLICT (date) DO UPDATE 
            SET actual_price = {actual_price}, predicted_price = {predicted_price}
        """)

def insert_forecast(cur, forecast):
    logging.info("Inserting forecast")
    # Convert forecast to list if it's a numpy array
    if isinstance(forecast, np.ndarray):
        forecast = forecast.tolist()

    # SQL query to insert or update forecast in the database
    query = """
        INSERT INTO forecasted_prices (date, forecasted_price) 
        VALUES (%s, %s) 
        ON CONFLICT (date) 
        DO UPDATE SET forecasted_price = EXCLUDED.forecasted_price
    """

    # Get today's date
    today = datetime.today()

    # Insert or update each forecasted price in the database
    for i, price in enumerate(forecast):
        date = today + timedelta(days=i+1)  # The date is the current date plus the forecast horizon
        cur.execute(query, (date, price))


def insert_evaluation_results(cur, train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2):
    logging.info("Inserting evaluation results")
    # SQL query to insert evaluation results into the database
    query = """
        INSERT INTO evaluation_results (train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.execute(query, (train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2))

def close_connection(conn):
    logging.info("Closing connection")
    # Close the database connection
    conn.close()
