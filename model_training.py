# model_training.py
import logging
import pandas as pd
from autots import AutoTS
from config import autots_params, initial_training, evolve, template_filename

logging.basicConfig(level=logging.DEBUG)

def train_model(data, forecast_length, future_regressor=None):
    # Define variables here so they exist in the local scope
    model = prediction = evaluation = prediction_df = None

    try:
        logging.info('Splitting data into training and test sets')
        split = int(0.8 * len(data))
        train_data = data[:split]
        test_data = data[split:]

        logging.info('Creating dataset for training and testing')
        x_train, y_train = [], []
        x_test, y_test = [], []
        for i in range(forecast_length, len(train_data)):
            x_train.append(train_data.iloc[i - forecast_length:i, 1:-1].values.flatten())
            y_train.append(train_data.iloc[i, -1])
        for i in range(forecast_length, len(test_data)):
            x_test.append(test_data.iloc[i - forecast_length:i, 1:-1].values.flatten())
            y_test.append(test_data.iloc[i, -1])

        logging.info('Converting datasets to pandas dataframes')
        x_train_pd = pd.DataFrame(x_train)
        y_train_pd = pd.DataFrame(y_train, columns=['Close'])
        x_test_pd = pd.DataFrame(x_test)
        y_test_pd = pd.DataFrame(y_test, columns=['Close'])

        logging.info('Preparing data for model training')
        train_data.reset_index(level=0, inplace=True)
        train_data.columns = ['date'] + list(train_data.columns)[1:]
        if train_data['date'].dtype != 'datetime64[ns]':
            train_data['date'] = pd.to_datetime(train_data['date'])
        test_data.reset_index(level=0, inplace=True)
        test_data.columns = ['date'] + list(test_data.columns)[1:]
        if test_data['date'].dtype != 'datetime64[ns]':
            test_data['date'] = pd.to_datetime(test_data['date'])

        logging.info('Adding Close and date to training and testing datasets')
        x_train_pd = pd.concat([train_data['Close'][forecast_length:].reset_index(drop=True), x_train_pd], axis=1)
        x_train_pd['date'] = train_data['date'][forecast_length:].reset_index(drop=True)
        x_test_pd = pd.concat([test_data['Close'][forecast_length:].reset_index(drop=True), x_test_pd], axis=1)
        x_test_pd['date'] = test_data['date'][forecast_length:].reset_index(drop=True)

    except Exception as e:
        logging.error(f"Error during data preparation: {e}")
        raise

    try:
        logging.info('Initializing AutoTS model')
        model = AutoTS(**autots_params)

        if initial_training:
            logging.info('Training model for the first time')
            model = model.fit(x_train_pd, date_col='date', value_col='Close', id_col=None)
            model.export_template(template_filename, models='best', n=15, max_per_model_class=3)
            logging.info('Model trained and template exported for the first time')
        else:
            try:
                if evolve:
                    logging.info('Evolving model')
                    model = model.import_template(template_filename, method='only', enforce_model_list=True)
                    model = model.fit(x_train_pd, date_col='date', value_col='Close', id_col=None)
                    model.export_template(template_filename, models='best', n=15, max_per_model_class=3)
                    logging.info('Model evolved and new template exported')
                else:
                    logging.info('Importing existing model template')
                    model = model.import_template(template_filename, method='only', enforce_model_list=True)
                    logging.info('Model imported from existing template')
            except FileNotFoundError:
                logging.warning("Couldn't find %s, doing initial training.", template_filename)
                logging.warning(f"Couldn't find {template_filename}, doing initial training.")
                model = model.fit(x_train_pd, date_col='date', value_col='Close', id_col=None)
                model.export_template(template_filename, models='best', n=15, max_per_model_class=3)
                logging.info('Model trained and template exported after FileNotFoundError')
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

    try:
        logging.info('Getting evaluation results')
        evaluation = model.results().to_dict('records')

        logging.info('Predicting future data')
        prediction = model.predict(forecast_length)

        logging.info('Creating prediction dataframe')
        prediction_df = prediction.forecast
    except Exception as e:
        logging.error(f"Error during model prediction: {e}")
        raise

    try:
        logging.info('Appending predicted prices to training and testing datasets')
        x_train_pd = pd.concat([x_train_pd, prediction_df], axis=1)
        x_test_pd = pd.concat([x_test_pd, prediction_df], axis=1)
    except Exception as e:
        logging.error(f"Error during post-prediction processing: {e}")
        raise

    return model, prediction, x_train_pd, x_test_pd, y_train_pd, y_test_pd, evaluation, prediction_df
