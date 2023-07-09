import logging  # Import the logging module
import h2o
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML
import config

# Initialize the H2O cluster
h2o.init()

# Access the parameters
use_kfold = config.use_kfold
kfold_splits = config.kfold_splits
early_stopping_patience = config.early_stopping_patience

def train_model(X_train, Y_train, X_test, Y_test, look_back, num_features, model_params):
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
    print("Target column name: ", y)
    print("List of column names: ", x)

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

    return model, preds
