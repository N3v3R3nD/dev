import logging  # Import the logging module
import h2o
from h2o.automl import H2OAutoML
import config

# Initialize the H2O cluster
h2o.init()

# Access the parameters
use_kfold = config.use_kfold
kfold_splits = config.kfold_splits
early_stopping_patience = config.early_stopping_patience

def train_model(X_train, Y_train, X_test, Y_test, look_back, num_features, model_params):
    # Convert data to H2O data frames
    X_train_h2o = h2o.H2OFrame(X_train.values.tolist() if isinstance(X_train, pd.DataFrame) else X_train.tolist())
    Y_train_h2o = h2o.H2OFrame(Y_train.values.tolist() if isinstance(Y_train, pd.DataFrame) else Y_train.tolist())
    X_test_h2o = h2o.H2OFrame(X_test.values.tolist() if isinstance(X_test, pd.DataFrame) else X_test.tolist())
    Y_test_h2o = h2o.H2OFrame(Y_test.values.tolist() if isinstance(Y_test, pd.DataFrame) else Y_test.tolist())


    # Combine features and target into a single data frame
    train_data = X_train_h2o.cbind(Y_train_h2o)
    test_data = X_test_h2o.cbind(Y_test_h2o)

    # Define the column names
    x = train_data.columns
    y = "target"  # Replace with the name of your target column
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
