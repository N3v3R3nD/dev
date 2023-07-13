# db_operation.py
import logging
from datetime import datetime, timedelta

import pandas as pd
import psycopg2

import config
import numpy as np
import json

# Extract database credentials from config
db_config = config.database
host = db_config['host']
database = db_config['database']
user = db_config['user']
password = db_config['password']


def connect_to_db():
    try:
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
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        raise

def create_tables(cur):
    try:
        # Create actual_vs_predicted table if it doesn't exist
        logging.info('Creating actual_vs_predicted table if it does not exist')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS actual_vs_predicted (
                execution_id SERIAL,
                date DATE,
                actual_price FLOAT,
                predicted_price FLOAT,
                PRIMARY KEY (execution_id, date)
            )
        """)
          # Create evaluation_results table if it doesn't exist
        logging.info('Creating evaluation_results table if it doesn\'t exist')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                execution_id SERIAL,
                ID TEXT,
                Model TEXT,
                ModelParameters TEXT,
                TransformationParameters TEXT,
                TransformationRuntime TEXT,
                FitRuntime TEXT,
                PredictRuntime TEXT,
                TotalRuntime TEXT,
                Ensemble TEXT,
                Exceptions TEXT,
                Runs INTEGER,
                Generation TEXT,
                ValidationRound INTEGER,
                ValidationStartDate TEXT,
                smape FLOAT,
                mae FLOAT,
                rmse FLOAT,
                made FLOAT,
                mage FLOAT,
                underestimate FLOAT,
                mle FLOAT,
                overestimate FLOAT,
                imle FLOAT,
                spl FLOAT,
                containment FLOAT,
                contour FLOAT,
                maxe FLOAT,
                oda FLOAT,
                dwae FLOAT,
                mqae FLOAT,
                ewmae FLOAT,
                uwmse FLOAT,
                smoothness FLOAT,
                smape_weighted FLOAT,
                mae_weighted FLOAT,
                rmse_weighted FLOAT,
                made_weighted FLOAT,
                mage_weighted FLOAT,
                underestimate_weighted FLOAT,
                mle_weighted FLOAT,
                overestimate_weighted FLOAT,
                imle_weighted FLOAT,
                spl_weighted FLOAT,
                containment_weighted FLOAT,
                contour_weighted FLOAT,
                maxe_weighted FLOAT,
                oda_weighted FLOAT,
                dwae_weighted FLOAT,
                mqae_weighted FLOAT,
                ewmae_weighted FLOAT,
                uwmse_weighted FLOAT,
                smoothness_weighted FLOAT,
                TotalRuntimeSeconds FLOAT,
                Score FLOAT,
                PRIMARY KEY (execution_id, ID)
            )
        """)
        # Create fetched_data table if it doesn't exist
        logging.info('Creating fetched_data table if it does not exist')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fetched_data (
                execution_id SERIAL,
                date DATE,
                open_price FLOAT,
                high_price FLOAT,
                low_price FLOAT,
                close_price FLOAT,
                adj_close_price FLOAT,
                volume FLOAT,
                PRIMARY KEY (execution_id, date)
            )
        """)

        # Create forecasted_prices table if it doesn't exist
        logging.info('Creating forecasted_prices table if it doesn\'t exist')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forecasted_prices (
                execution_id SERIAL,
                date DATE,
                forecasted_price FLOAT,
                PRIMARY KEY (execution_id, date)
            )
        """)

        # Create execution_settings table if it doesn't exist
        logging.info('Creating execution_settings table if it doesn\'t exist')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS execution_settings (
                execution_id SERIAL PRIMARY KEY,
                config_settings TEXT,
                model_parameters TEXT
            )
        """)

    except Exception as e:
        logging.error(f"Error creating tables: {e}")
        raise


def get_next_execution_id(cur):
    try:
        # Create fetched_data table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fetched_data (
                execution_id SERIAL,
                date DATE,
                open_price FLOAT,
                high_price FLOAT,
                low_price FLOAT,
                close_price FLOAT,
                adj_close_price FLOAT,
                volume FLOAT,
                PRIMARY KEY (execution_id, date)
            )
        """)

        # Fetch the last execution_id from the fetched_data table
        cur.execute("SELECT MAX(execution_id) FROM fetched_data")
        last_execution_id = cur.fetchone()[0]

        # If this is the first execution, start from 1, otherwise increment the last execution_id by 1
        execution_id = 1 if last_execution_id is None else last_execution_id + 1

        return execution_id
    except Exception as e:
        logging.error(f"Error fetching next execution id: {e}")
        raise


def insert_execution_settings(cur, execution_id, config, model):
    try:
        logging.info('Inserting execution settings into the database')

        # Create a new dictionary with only the necessary settings
        config_dict = {
            'forecast_steps': config.forecast_steps,
            'database': config.database,
            # Add any other settings you need
        }

        # Convert config dictionary and model parameters to string
        config_settings = json.dumps(config_dict)
        model_parameters = json.dumps(model.params)

        # SQL query to insert execution settings into the database
        query = """
            INSERT INTO execution_settings (execution_id, config_settings, model_parameters) 
            VALUES (%s, %s, %s)
        """
        cur.execute(query, (execution_id, config_settings, model_parameters))
    except Exception as e:
        logging.error(f"Error inserting execution settings: {e}")
        raise

def insert_data(cur, execution_id, data, forecast):
    try:
        logging.info('Inserting actual and predicted prices into the database')

        # Insert actual and predicted prices into the database
        for i in range(len(data)-1):  # Exclude the last record as it does not have a prediction
            date = data.index[i].strftime('%Y-%m-%d')
            actual_price = data.iloc[i]['Close']
            predicted_price = data.iloc[i]['Predicted'] if 'Predicted' in data.columns else None

            if predicted_price is not None:
                cur.execute(f'''
                    INSERT INTO actual_vs_predicted (execution_id, date, actual_price, predicted_price) 
                    VALUES ({execution_id}, '{date}', {actual_price}, {predicted_price}) 
                    ON CONFLICT (execution_id, date) DO UPDATE 
                    SET actual_price = {actual_price}, predicted_price = {predicted_price}
                ''')
            else:
                cur.execute(f'''
                    INSERT INTO actual_vs_predicted (execution_id, date, actual_price) 
                    VALUES ({execution_id}, '{date}', {actual_price}) 
                    ON CONFLICT (execution_id, date) DO UPDATE 
                    SET actual_price = {actual_price}
                ''')

        # Determine the length of the loop
        loop_length = len(forecast.forecast)

        # Insert forecasted prices into the database
        logging.info('Inserting forecasted prices into the database')

        for i in range(loop_length):  
            date = (data.index[-1] + pd.DateOffset(days=i+1)).strftime('%Y-%m-%d')  # The date is the last date in data plus the forecast horizon
            forecasted_price = forecast.forecast.iloc[i][0]  # Assuming forecast is a DataFrame with forecasted prices

            cur.execute(f'''
                INSERT INTO forecasted_prices (execution_id, date, forecasted_price) 
                VALUES ({execution_id}, '{date}', {forecasted_price}) 
                ON CONFLICT (execution_id, date) DO UPDATE 
                SET forecasted_price = {forecasted_price}
            ''')
    except Exception as e:
        logging.error(f'Error inserting data: {e}')
        raise

def insert_evaluation_results(cur, execution_id, evaluation):
    try:
        query = """
            INSERT INTO evaluation_results (
                execution_id,
                ID,
                Model,
                ModelParameters,
                TransformationParameters,
                TransformationRuntime,
                FitRuntime,
                PredictRuntime,
                TotalRuntime,
                Ensemble,
                Exceptions,
                Runs,
                Generation,
                ValidationRound,
                ValidationStartDate,
                smape,
                mae,
                rmse,
                made,
                mage,
                underestimate,
                mle,
                overestimate,
                imle,
                spl,
                containment,
                contour,
                maxe,
                oda,
                dwae,
                mqae,
                ewmae,
                uwmse,
                smoothness,
                smape_weighted,
                mae_weighted,
                rmse_weighted,
                made_weighted,
                mage_weighted,
                underestimate_weighted,
                mle_weighted,
                overestimate_weighted,
                imle_weighted,
                spl_weighted,
                containment_weighted,
                contour_weighted,
                maxe_weighted,
                oda_weighted,
                dwae_weighted,
                mqae_weighted,
                ewmae_weighted,
                uwmse_weighted,
                smoothness_weighted,
                TotalRuntimeSeconds,
                Score
            )
            VALUES (
                %(execution_id)s,
                %(ID)s,
                %(Model)s,
                %(ModelParameters)s,
                %(TransformationParameters)s,
                %(TransformationRuntime)s,
                %(FitRuntime)s,
                %(PredictRuntime)s,
                %(TotalRuntime)s,
                %(Ensemble)s,
                %(Exceptions)s,
                %(Runs)s,
                %(Generation)s,
                %(ValidationRound)s,
                %(ValidationStartDate)s,
                %(smape)s,
                %(mae)s,
                %(rmse)s,
                %(made)s,
                %(mage)s,
                %(underestimate)s,
                %(mle)s,
                %(overestimate)s,
                %(imle)s,
                %(spl)s,
                %(containment)s,
                %(contour)s,
                %(maxe)s,
                %(oda)s,
                %(dwae)s,
                %(mqae)s,
                %(ewmae)s,
                %(uwmse)s,
                %(smoothness)s,
                %(smape_weighted)s,
                %(mae_weighted)s,
                %(rmse_weighted)s,
                %(made_weighted)s,
                %(mage_weighted)s,
                %(underestimate_weighted)s,
                %(mle_weighted)s,
                %(overestimate_weighted)s,
                %(imle_weighted)s,
                %(spl_weighted)s,
                %(containment_weighted)s,
                %(contour_weighted)s,
                %(maxe_weighted)s,
                %(oda_weighted)s,
                %(dwae_weighted)s,
                %(mqae_weighted)s,
                %(ewmae_weighted)s,
                %(uwmse_weighted)s,
                %(smoothness_weighted)s,
                %(TotalRuntimeSeconds)s,
                %(Score)s
            )
            ON CONFLICT (execution_id, ID) DO UPDATE SET
                (Model, ModelParameters, TransformationParameters, TransformationRuntime, FitRuntime, PredictRuntime,
                TotalRuntime, Ensemble, Exceptions, Runs, Generation, ValidationRound, ValidationStartDate, smape,
                mae, rmse, made, mage, underestimate, mle, overestimate, imle, spl, containment, contour, maxe,
                oda, dwae, mqae, ewmae, uwmse, smoothness, smape_weighted, mae_weighted, rmse_weighted, made_weighted,
                mage_weighted, underestimate_weighted, mle_weighted, overestimate_weighted, imle_weighted, spl_weighted,
                containment_weighted, contour_weighted, maxe_weighted, oda_weighted, dwae_weighted, mqae_weighted,
                ewmae_weighted, uwmse_weighted, smoothness_weighted, TotalRuntimeSeconds, Score) =
                (EXCLUDED.Model, EXCLUDED.ModelParameters, EXCLUDED.TransformationParameters, EXCLUDED.TransformationRuntime, EXCLUDED.FitRuntime,
                EXCLUDED.PredictRuntime, EXCLUDED.TotalRuntime, EXCLUDED.Ensemble, EXCLUDED.Exceptions, EXCLUDED.Runs, EXCLUDED.Generation,
                EXCLUDED.ValidationRound, EXCLUDED.ValidationStartDate, EXCLUDED.smape, EXCLUDED.mae, EXCLUDED.rmse, EXCLUDED.made, EXCLUDED.mage,
                EXCLUDED.underestimate, EXCLUDED.mle, EXCLUDED.overestimate, EXCLUDED.imle, EXCLUDED.spl, EXCLUDED.containment, EXCLUDED.contour,
                EXCLUDED.maxe, EXCLUDED.oda, EXCLUDED.dwae, EXCLUDED.mqae, EXCLUDED.ewmae, EXCLUDED.uwmse, EXCLUDED.smoothness,
                EXCLUDED.smape_weighted, EXCLUDED.mae_weighted, EXCLUDED.rmse_weighted, EXCLUDED.made_weighted, EXCLUDED.mage_weighted,
                EXCLUDED.underestimate_weighted, EXCLUDED.mle_weighted, EXCLUDED.overestimate_weighted, EXCLUDED.imle_weighted, EXCLUDED.spl_weighted,
                EXCLUDED.containment_weighted, EXCLUDED.contour_weighted, EXCLUDED.maxe_weighted, EXCLUDED.oda_weighted, EXCLUDED.dwae_weighted,
                EXCLUDED.mqae_weighted, EXCLUDED.ewmae_weighted, EXCLUDED.uwmse_weighted, EXCLUDED.smoothness_weighted, EXCLUDED.TotalRuntimeSeconds,
                EXCLUDED.Score)
                    """
        values = []  # Define values here
        for row in evaluation:
            # Add execution_id to each row
            row['execution_id'] = execution_id

            # Explicitly convert "NaT" to None for 'ValidationStartDate'
            if row['ValidationStartDate'] == "NaT" or pd.isnull(row['ValidationStartDate']):
                row['ValidationStartDate'] = None

            values.append(row)

        cur.executemany(query, values)
    except Exception as e:
        logging.error(f"Error inserting evaluation results: {e}")
        raise



def insert_forecast(cur, execution_id, forecast, target_scaler):
    try:
        logging.info("Inserting forecast")
        # Convert forecast to list if it's a numpy array
        if isinstance(forecast, np.ndarray):
            forecast = forecast.tolist()

        # Get today's date
        today = datetime.today()

        # SQL query to insert or update forecast in the database
        query = """
            INSERT INTO forecasted_prices (execution_id, date, forecasted_price) 
            VALUES (%s, %s, %s) 
            ON CONFLICT (execution_id, date) 
            DO UPDATE SET forecasted_price = EXCLUDED.forecasted_price
        """
        # Insert or update each forecasted price in the database
        for i, price in enumerate(forecast):
            date = today + timedelta(days=i+1)  # The date is the current date plus the forecast horizon

            # Inverse transform the forecasted price
            forecasted_price = target_scaler.inverse_transform(np.array([[price]]))[0][0]

            cur.execute(query, (execution_id, date, forecasted_price)) # added execution_id here
    except Exception as e:
        logging.error(f"Error inserting forecast: {e}")
        raise

def insert_fetched_data(cur, execution_id, data):
    try:
        logging.info('Inserting fetched data into the database')
        for i in range(len(data)):
            date = data.index[i].strftime('%Y-%m-%d')  # Get the date as a string
            open_price = data['Open'].iloc[i]  # Get the open price
            high_price = data['High'].iloc[i]  # Get the high price
            low_price = data['Low'].iloc[i]  # Get the low price
            close_price = data['Close'].iloc[i]  # Get the close price
            adj_close_price = data['Adj Close'].iloc[i]  # Get the adjusted close price
            volume = int(data['Volume'].iloc[i])  # Get the volume and convert it to a native Python int

            cur.execute("""
                INSERT INTO fetched_data (execution_id, date, open_price, high_price, low_price, close_price, adj_close_price, volume) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
                ON CONFLICT (execution_id, date) DO UPDATE 
                SET open_price = %s, high_price = %s, low_price = %s, close_price = %s, adj_close_price = %s, volume = %s
            """, (execution_id, date, open_price, high_price, low_price, close_price, adj_close_price, volume, 
                open_price, high_price, low_price, close_price, adj_close_price, volume))
    except Exception as e:
        logging.error(f"Error inserting fetched data: {e}")
        raise

def close_connection(conn):
    try:
        logging.info("Closing connection")
        # Close the database connection
        conn.close()
    except Exception as e:
        logging.error(f"Error closing connection: {e}")
        raise