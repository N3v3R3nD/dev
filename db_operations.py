import psycopg2
from datetime import datetime, timedelta
import logging
import config

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
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

def insert_data(cur, Y_train, train_predict, test_predict, target_scaler):    
    # Insert actual and predicted prices into the database
    logging.info('Inserting actual and predicted prices into the database')

    if len(Y_train) != len(train_predict):
        raise ValueError("Length of Y_train and train_predict don't match")

    for i in range(len(Y_train)):
        date = (datetime.today() - timedelta(days=len(Y_train) - i - 1)).strftime('%Y-%m-%d')  # Calculate the correct date
        actual_price = target_scaler.inverse_transform(Y_train[i].reshape(-1, 1))[0][0]
        predicted_price = train_predict[i][0]
        
        cur.execute(f"""
            INSERT INTO actual_vs_predicted (date, actual_price, predicted_price) 
            VALUES ('{date}', {actual_price}, {predicted_price}) 
            ON CONFLICT (date) DO UPDATE 
            SET actual_price = {actual_price}, predicted_price = {predicted_price}
        """)

def insert_evaluation_results(cur, train_rmse, test_rmse, train_mae, test_mae):
    # Insert evaluation results into the database
    logging.info('Inserting evaluation results into the database')
    cur.execute(f"""
        INSERT INTO evaluation_results (train_rmse, test_rmse, train_mae, test_mae) 
        VALUES ({train_rmse}, {test_rmse}, {train_mae}, {test_mae})
    """)

def close_connection(conn):
    # Commit changes and close connection
    conn.commit()
    conn.close()
