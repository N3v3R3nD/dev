# model_training.py
import logging
import numpy as np
import pandas as pd
from autots import AutoTS
from config import autots_params
import pickle
logging.basicConfig(level=logging.DEBUG)
from config import autots_params

def train_model(data, forecast_length):
    logging.info('Splitting data into training and test sets')
    split = int(0.8 * len(data))
    train_data = data[:split]
    test_data = data[split:]

    # Log the shape and first few rows of train_data and test_data
    logging.debug(f"Shape of train_data: {train_data.shape}")
    logging.debug(f"First few rows of train_data:\n{train_data.head()}")
    logging.debug(f"Shape of test_data: {test_data.shape}")
    logging.debug(f"First few rows of test_data:\n{test_data.head()}")

    # Create the dataset for training
    logging.info('Creating dataset for training')
    X_train, Y_train = [], []
    for i in range(forecast_length, len(train_data)):
        X_train.append(train_data.iloc[i - forecast_length:i, 1:-1].values.flatten())
        Y_train.append(train_data.iloc[i, -1])

    # Create the dataset for testing
    logging.info('Creating dataset for testing')
    X_test, Y_test = [], []
    for i in range(forecast_length, len(test_data)):
        X_test.append(test_data.iloc[i - forecast_length:i, 1:-1].values.flatten())
        Y_test.append(test_data.iloc[i, -1])

    # Convert lists to pandas dataframes
    X_train_pd = pd.DataFrame(X_train)
    Y_train_pd = pd.DataFrame(Y_train, columns=['Close'])
    X_test_pd = pd.DataFrame(X_test)
    Y_test_pd = pd.DataFrame(Y_test, columns=['Close'])

    # Train model
    train_data.reset_index(level=0, inplace=True)
    train_data.columns = ['date'] + list(train_data.columns)[1:]
    if train_data['date'].dtype != 'datetime64[ns]':
        train_data['date'] = pd.to_datetime(train_data['date'])

    # Do the same for test_data
    test_data.reset_index(level=0, inplace=True)
    test_data.columns = ['date'] + list(test_data.columns)[1:]
    if test_data['date'].dtype != 'datetime64[ns]':
        test_data['date'] = pd.to_datetime(test_data['date'])

    # Add 'Close' and 'date' to X_train_pd and X_test_pd
    X_train_pd = pd.concat([train_data['Close'][forecast_length:].reset_index(drop=True), X_train_pd], axis=1)
    X_train_pd['date'] = train_data['date'][forecast_length:].reset_index(drop=True)
    X_test_pd = pd.concat([test_data['Close'][forecast_length:].reset_index(drop=True), X_test_pd], axis=1)
    X_test_pd['date'] = test_data['date'][forecast_length:].reset_index(drop=True)

    model = AutoTS(**autots_params)
    model = model.fit(X_train_pd, date_col='date', value_col='Close', id_col=None)

    # Get the evaluation results
    evaluation = model.results()
    # Convert the DataFrame to a list of dictionaries
    evaluation = evaluation.to_dict('records')

    # Make predictions
    logging.info("Starting prediction")
    prediction = model.predict(forecast_length)
    logging.info("Prediction completed")

    # Log the prediction
    logging.debug(f"Prediction: {prediction}")
    
    # Get the predicted values
    prediction_values = prediction.forecast
    logging.debug(f"Prediction values: \n{prediction_values}")

    # Since prediction_values is already a DataFrame, we don't need to create another DataFrame
    prediction_df = prediction_values
    logging.debug(f"Created prediction_df with shape: {prediction_df.shape}")

    # Log the prediction_df
    logging.debug(f"Prediction DataFrame:\n{prediction_df}")

    # Append predicted prices to X_train_pd and X_test_pd
    X_train_pd = pd.concat([X_train_pd, prediction_df], axis=1)
    X_test_pd = pd.concat([X_test_pd, prediction_df], axis=1)
    logging.debug(f"prediction_df in train_model: \n{prediction_df}")


    return model, prediction, X_train_pd, X_test_pd, Y_train_pd, Y_test_pd, evaluation, prediction_df

