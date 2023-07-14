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
    """Establishes a connection to the database and returns a cursor."""
    try:
        logging.info('Connecting to the database')
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cur = conn.cursor()
        logging.info('Successfully connected to the database')
        return conn, cur
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        raise


def create_tables(cur):
    """Creates necessary tables in the database if they do not exist."""
    try:
        logging.info('Creating tables if they do not exist')

        # List of SQL commands to create tables
        commands = [
            """
            CREATE TABLE IF NOT EXISTS actual_vs_predicted (
                execution_id SERIAL,
                date DATE,
                actual_price FLOAT,
                predicted_price FLOAT,
                PRIMARY KEY (execution_id, date)
            )
            """,
            """
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
            """,
            """
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
            """,
            """
            CREATE TABLE IF NOT EXISTS forecasted_prices (
                execution_id SERIAL,
                date DATE,
                forecasted_price FLOAT,
                upper_forecast FLOAT,
                lower_forecast FLOAT,
                model_parameters TEXT,
                transformation_parameters TEXT,
                forecast_length INT,
                model_name TEXT,
                avg_metrics FLOAT,
                avg_metrics_weighted FLOAT,
                per_series_metrics TEXT,
                per_timestamp TEXT,
                fit_runtime TIME,
                predict_runtime TIME,
                total_runtime TIME,
                transformation_runtime TIME,
                PRIMARY KEY (execution_id, date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS execution_settings (
                execution_id SERIAL PRIMARY KEY,
                config_settings TEXT,
                model_parameters TEXT
            )
            """
        ]

        # Execute each command
        for command in commands:
            cur.execute(command)

        logging.info('Successfully created tables')
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


def insert_data(cur, execution_id, data, prediction_df):
    try:
        logging.info('Inserting data into the database')

        # Insert actual and predicted prices into the database
        for i in range(len(data)):  
            date = data.index[i].strftime('%Y-%m-%d')
            actual_price = data.iloc[i]['Close']
            predicted_price = prediction_df.iloc[i]['Close'] if i < len(prediction_df) else None

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

    except Exception as e:
        logging.error(f'Error inserting data: {e}')
        raise



def insert_forecast(cur, execution_id, prediction):
    try:
        logging.info("Inserting forecast")
        logging.debug(f"Type of prediction object: {type(prediction)}")
        logging.debug(f"Attributes of prediction object: {dir(prediction)}")

        # Get today's date
        today = datetime.today()

        # SQL query to insert or update forecast in the database
        query = """
            INSERT INTO forecasted_prices (
                execution_id, 
                date, 
                forecasted_price,
                upper_forecast,
                lower_forecast,
                model_parameters,
                transformation_parameters,
                forecast_length,
                model_name,
                avg_metrics,
                avg_metrics_weighted,
                per_series_metrics,
                per_timestamp,
                fit_runtime,
                predict_runtime,
                total_runtime,
                transformation_runtime) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
            ON CONFLICT (execution_id, date) 
            DO UPDATE SET 
                forecasted_price = EXCLUDED.forecasted_price,
                upper_forecast = EXCLUDED.upper_forecast,
                lower_forecast = EXCLUDED.lower_forecast,
                model_parameters = EXCLUDED.model_parameters,
                transformation_parameters = EXCLUDED.transformation_parameters,
                forecast_length = EXCLUDED.forecast_length,
                model_name = EXCLUDED.model_name,
                avg_metrics = EXCLUDED.avg_metrics,
                avg_metrics_weighted = EXCLUDED.avg_metrics_weighted,
                per_series_metrics = EXCLUDED.per_series_metrics,
                per_timestamp = EXCLUDED.per_timestamp,
                fit_runtime = EXCLUDED.fit_runtime,
                predict_runtime = EXCLUDED.predict_runtime,
                total_runtime = EXCLUDED.total_runtime,
                transformation_runtime = EXCLUDED.transformation_runtime
        """
        for i in range(len(prediction.forecast)):
            date = (today + timedelta(days=i+1)).strftime('%Y-%m-%d')
            forecasted_price = prediction.forecast.iloc[i][0]
            upper_forecast = prediction.upper_forecast.iloc[i][0]
            lower_forecast = prediction.lower_forecast.iloc[i][0]
            model_parameters = str(prediction.model_parameters)
            transformation_parameters = str(prediction.transformation_parameters)
            forecast_length = prediction.forecast_length
            model_name = prediction.model_name
            avg_metrics = prediction.avg_metrics
            avg_metrics_weighted = prediction.avg_metrics_weighted
            per_series_metrics = prediction.per_series_metrics
            per_timestamp = prediction.per_timestamp
            fit_runtime = prediction.fit_runtime
            predict_runtime = prediction.predict_runtime
            total_runtime = prediction.total_runtime() if callable(prediction.total_runtime) else prediction.total_runtime
            transformation_runtime = prediction.transformation_runtime
            logging.debug("Type of forecasted_price: %s", type(forecasted_price))
            logging.debug("Type of upper_forecast: %s", type(upper_forecast))
            logging.debug("Type of lower_forecast: %s", type(lower_forecast))
            logging.debug("Type of model_parameters: %s", type(model_parameters))
            logging.debug("Type of transformation_parameters: %s", type(transformation_parameters))
            logging.debug("Type of forecast_length: %s", type(forecast_length))
            logging.debug("Type of model_name: %s", type(model_name))
            logging.debug("Type of avg_metrics: %s", type(avg_metrics))
            logging.debug("Type of avg_metrics_weighted: %s", type(avg_metrics_weighted))
            logging.debug("Type of per_series_metrics: %s", type(per_series_metrics))
            logging.debug("Type of per_timestamp: %s", type(per_timestamp))
            logging.debug("Type of fit_runtime: %s", type(fit_runtime))
            logging.debug("Type of predict_runtime: %s", type(predict_runtime))
            logging.debug("Type of total_runtime: %s", type(total_runtime))
            logging.debug("Type of transformation_runtime: %s", type(transformation_runtime))
            cur.execute(query, (execution_id, date, forecasted_price, upper_forecast, lower_forecast, model_parameters, transformation_parameters, forecast_length, model_name, avg_metrics, avg_metrics_weighted, per_series_metrics, per_timestamp, fit_runtime, predict_runtime, total_runtime, transformation_runtime))
    except Exception as e:
        logging.error(f'Error inserting forecast: {e}')
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
