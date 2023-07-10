import logging
from datetime import datetime, timedelta

import psycopg2

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
    # Create fetched_data table if it doesn't exist
    logging.info('Creating fetched_data table if it does not exist')
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fetched_data (
            date DATE PRIMARY KEY,
            open_price FLOAT,
            high_price FLOAT,
            low_price FLOAT,
            close_price FLOAT,
            adj_close_price FLOAT,
            volume FLOAT
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



def insert_data(cur, Y_train, train_preds, forecast, target_scaler):
    # Insert actual and predicted prices into the database
    logging.info('Inserting actual and predicted prices into the database')

    if Y_train.shape[0] != train_preds.shape[0]:
        raise ValueError("Length of Y_train and train_preds don't match")

    for i, actual_price in enumerate(Y_train):
        date = (datetime.today() - timedelta(days=len(Y_train) - i - 1)).strftime('%Y-%m-%d')  # Calculate the correct date
        actual_price = target_scaler.inverse_transform(actual_price.reshape(-1, 1))[0][0]
        predicted_price = target_scaler.inverse_transform(train_preds[i].reshape(-1, 1))[0][0]

        cur.execute(f"""
            INSERT INTO actual_vs_predicted (date, actual_price, predicted_price) 
            VALUES ('{date}', {actual_price}, {predicted_price}) 
            ON CONFLICT (date) DO UPDATE 
            SET actual_price = {actual_price}, predicted_price = {predicted_price}
        """)



def insert_forecast(cur, forecast, target_scaler):
    logging.info("Inserting forecast")
    # Convert forecast to list if it's a numpy array
    if isinstance(forecast, np.ndarray):
        forecast = forecast.tolist()

    # Get today's date
    today = datetime.today()

    # SQL query to insert or update forecast in the database
    query = """
        INSERT INTO forecasted_prices (date, forecasted_price) 
        VALUES (%s, %s) 
        ON CONFLICT (date) 
        DO UPDATE SET forecasted_price = EXCLUDED.forecasted_price
    """
    # Insert or update each forecasted price in the database
    for i, price in enumerate(forecast):
        date = today + timedelta(days=i+1)  # The date is the current date plus the forecast horizon

        # Inverse transform the forecasted price
        forecasted_price = target_scaler.inverse_transform(np.array([[price]]))[0][0]

        cur.execute(query, (date, forecasted_price))


def insert_fetched_data(cur, data):
    logging.info('Inserting fetched data into the database')
    for i in range(len(data)):
        date = data.index[i].strftime('%Y-%m-%d')  # Get the date as a string
        open_price = data['Open'].iloc[i]  # Get the open price
        high_price = data['High'].iloc[i]  # Get the high price
        low_price = data['Low'].iloc[i]  # Get the low price
        close_price = data['Close'].iloc[i]  # Get the close price
        adj_close_price = data['Adj Close'].iloc[i]  # Get the adjusted close price
        volume = data['Volume'].iloc[i]  # Get the volume

        cur.execute(f"""
            INSERT INTO fetched_data (date, open_price, high_price, low_price, close_price, adj_close_price, volume) 
            VALUES ('{date}', {open_price}, {high_price}, {low_price}, {close_price}, {adj_close_price}, {volume}) 
            ON CONFLICT (date) DO UPDATE 
            SET open_price = {open_price}, high_price = {high_price}, low_price = {low_price}, close_price = {close_price}, adj_close_price = {adj_close_price}, volume = {volume}
        """)


def insert_evaluation_results(cur, train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2):
    logging.info("Inserting evaluation results")
    # SQL query to insert evaluation results into the database
    query = """
        INSERT INTO evaluation_results (train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.execute(query, (train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2))
    logging.debug(f"Inserted evaluation results")

def close_connection(conn):
    logging.info("Closing connection")
    # Close the database connection
    conn.close()
