import logging 
import h2o
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML
import config

# Initialize the H2O cluster
h2o.init()

forecast_steps = config.forecast_steps

def train_model(X_train, Y_train, X_test, Y_test, forecast_steps, num_features, model_params):
    logging.info("Starting model training")
    # Reshape data if it's a 3D numpy array
    if isinstance(X_train, np.ndarray) and len(X_train.shape) == 3:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if isinstance(X_test, np.ndarray) and len(X_test.shape) == 3:
        X_test = X_test.reshape(X_test.shape[0], -1)

    # Convert data to H2O data frames
    X_train_h2o = h2o.H2OFrame(X_train if isinstance(X_train, pd.DataFrame) else X_train.tolist())
    Y_train_h2o = h2o.H2OFrame(Y_train if isinstance(Y_train, pd.DataFrame) else Y_train.tolist())
    X_test_h2o = h2o.H2OFrame(X_test if isinstance(X_test, pd.DataFrame) else X_test.tolist())
    Y_test_h2o = h2o.H2OFrame(Y_test if isinstance(Y_test, pd.DataFrame) else Y_test.tolist())

    # Combine features and target into a single data frame
    train_data = X_train_h2o.cbind(Y_train_h2o)
    test_data = X_test_h2o.cbind(Y_test_h2o)

    # Print out the target column name and the list of column names
    logging.debug("Target column name: ", y)
    logging.debug("List of column names: ", x)
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

    # Generate new input data for forecast
    forecast_input = X_test[-forecast_steps:]  # Get the most recent observations

    # Convert forecast input to H2O data frame
    forecast_input_h2o = h2o.H2OFrame(forecast_input if isinstance(forecast_input, pd.DataFrame) else forecast_input.tolist())

    # Make forecast
    forecast = model.predict(forecast_input_h2o)

    # Convert forecast to numpy array
    forecast = forecast.as_data_frame().values
    logging.info("Model training completed")
    return model, preds, forecast
