# model_training.py
import logging
import pandas as pd
from autots import AutoTS
from config import autots_params, initial_training, evolve, template_filename
logging.basicConfig(level=logging.DEBUG)

def train_model(data, forecast_length, future_regressor=None):
    model = prediction_df = None  # Add the initial definition here
    try:
        logging.info('Splitting data into training and test sets')
        split = int(0.8 * len(data))
        train_data = data[:split]
        test_data = data[split:]

        # Log the shape and first few rows of train_data and test_data
        logging.debug("Shape of train_data: %s", train_data.shape)
        logging.debug("First few rows of train_data:\n%s", train_data.head())
        logging.debug("Shape of test_data: %s", test_data.shape)
        logging.debug("First few rows of test_data:\n%s", test_data.head())

        # Create the dataset for training
        logging.info('Creating dataset for training')
        x_train, y_train = [], []
        for i in range(forecast_length, len(train_data)):
            x_train.append(train_data.iloc[i - forecast_length:i, 1:-1].values.flatten())
            y_train.append(train_data.iloc[i, -1])

        # Create the dataset for testing
        logging.info('Creating dataset for testing')
        x_test, y_test = [], []
        for i in range(forecast_length, len(test_data)):
            x_test.append(test_data.iloc[i - forecast_length:i, 1:-1].values.flatten())
            y_test.append(test_data.iloc[i, -1])

        # Convert lists to pandas dataframes
        x_train_pd = pd.DataFrame(x_train)
        y_train_pd = pd.DataFrame(y_train, columns=['Close'])
        x_test_pd = pd.DataFrame(x_test)
        y_test_pd = pd.DataFrame(y_test, columns=['Close'])

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

        # Add 'Close' and 'date' to x_train_pd and x_test_pd
        x_train_pd = pd.concat([train_data['Close'][forecast_length:].reset_index(drop=True), x_train_pd], axis=1)
        x_train_pd['date'] = train_data['date'][forecast_length:].reset_index(drop=True)
        x_test_pd = pd.concat([test_data['Close'][forecast_length:].reset_index(drop=True), x_test_pd], axis=1)
        x_test_pd['date'] = test_data['date'][forecast_length:].reset_index(drop=True)

        # Initialize the model
        model = AutoTS(**autots_params)
        logging.info("Initialized AutoTS model")

        if initial_training:
            # Fit the model
            logging.info("Starting initial training of the model")
            model = model.fit(
                x_train_pd,
                date_col='date',
                value_col='Close',
                id_col=None,
                future_regressor=future_regressor
            )
            # Export the model template
            model.export_template(template_filename, models='best', n=15, max_per_model_class=3)
            logging.info("Exported model template to %s after initial training", template_filename)
        elif evolve:
            # Import the existing model template
            model = model.import_template(template_filename, method='only', enforce_model_list=True)
            logging.info("Imported existing model template from %s for evolution", template_filename)

            # Continue training the model
            logging.info("Starting evolution of the model")
            model = model.fit(x_train_pd, date_col='date', value_col='Close', id_col=None)

            # Export the evolved model template
            model.export_template(template_filename, models='best', n=15, max_per_model_class=3)
            logging.info("Exported evolved model template to %s", template_filename)
        else:
            # Import the existing model template
            model = model.import_template(template_filename, method='only', enforce_model_list=True)
            logging.info("Imported existing model template from %s without evolution", template_filename)
    except Exception as error:
        logging.error("Error during model training: %s", error)

    try:
        # Get the evaluation results
        evaluation = model.results()
        # Convert the DataFrame to a list of dictionaries
        evaluation = evaluation.to_dict('records')

        # Make predictions
        logging.info("Starting prediction")
        prediction = model.predict(forecast_length)
        logging.info("Prediction completed")

        # Log the prediction
        logging.debug("Prediction: %s", prediction)
        
        # Get the predicted values
        prediction_values = prediction.forecast
        logging.debug("Prediction values: \n%s", prediction_values)

        # Since prediction_values is already a DataFrame, we don't need to create another DataFrame
        prediction_df = prediction_values
        logging.debug("Created prediction_df with shape: %s", prediction_df.shape)
    except Exception as error:
        logging.error("Error during prediction: %s", error)

    try:
        # Log the prediction_df
        logging.debug("Prediction DataFrame:\n%s", prediction_df)

        # Append predicted prices to x_train_pd and x_test_pd
        x_train_pd = pd.concat([x_train_pd, prediction_df], axis=1)
        x_test_pd = pd.concat([x_test_pd, prediction_df], axis=1)
        logging.debug("prediction_df in train_model: \n%s", prediction_df)
    except Exception as error:
        logging.error("Error during post-prediction processing: %s", error)

    return model, prediction, x_train_pd, x_test_pd, y_train_pd, y_test_pd, evaluation, prediction_df
