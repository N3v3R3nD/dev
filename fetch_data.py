# fetch_data.py
import logging
from datetime import datetime

import config
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')

def fetch_data():
    try:
        # Access the parameters
        yfinance_symbol = config.yfinance_symbol

        # Fetch data using yfinance
        logging.info('Fetching data using yfinance')
        data = yf.download(yfinance_symbol, start=config.start_date, end=config.end_date)
        logging.info('Data downloaded from Yahoo Finance')
        logging.debug('Data shape: %s', data.shape)
        logging.debug('First few rows of the data:')
        logging.debug(data.head())

        # Create a shifted version of the 'Open' column as the target
        data['Target'] = data['Open'].shift(-1)

        # Drop the last row, which does not have a target
        data = data[:-1]

        # Fetch macroeconomic data using FRED
        logging.info('Fetching macroeconomic data using FRED')
        gdp = pdr.get_data_fred('GDP', config.start_date, config.end_date)
        unemployment = pdr.get_data_fred('UNRATE', config.start_date, config.end_date)

        # Preprocess macroeconomic data
        logging.info('Preprocessing macroeconomic data')
        gdp = gdp.resample('D').interpolate()  # Fill missing values by interpolation
        unemployment = unemployment.resample('D').interpolate()  # Fill missing values by interpolation

        # Merge macroeconomic data with existing data
        logging.info('Merging macroeconomic data with existing data')
        data = pd.merge(data, gdp, how='left', left_index=True, right_index=True)
        data = pd.merge(data, unemployment, how='left', left_index=True, right_index=True)

        # Engineer features from macroeconomic data
        logging.info('Engineering features from macroeconomic data')
        data['GDP Change'] = data['GDP'].pct_change()
        data['Unemployment Change'] = data['UNRATE'].pct_change()

        # Drop original macroeconomic columns
        data = data.drop(columns=['GDP', 'UNRATE'])

        # Add day of the week feature
        data['DayOfWeek'] = data.index.dayofweek

        # Add month of the year feature
        data['Month'] = data.index.month

        # Add is holiday feature
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=data.index.min(), end=data.index.max())
        data['IsHoliday'] = data.index.isin(holidays).astype(int)

        # Filter out non-business days
        data = data[data.index.dayofweek < 5]

        # Add technical indicators
        data['SMA'] = data['Close'].rolling(window=14).mean()
        data['EMA'] = data['Close'].ewm(span=14).mean()
        delta = data['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        data['RSI'] = 100 - (100 / (1 + rs))
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        data['MACD'] = macd - signal
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['20dSTD'] = data['Close'].rolling(window=20).std()
        data['Upper'] = data['MA20'] + (data['20dSTD'] * 2)
        data['Lower'] = data['MA20'] - (data['20dSTD'] * 2)
        data['Cum_Daily_Returns'] = (data['Close'] / data['Close'].shift(1)) - 1
        data['Cumulative_Returns'] = (1 + data['Cum_Daily_Returns']).cumprod()
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        data = data.dropna()

        # Ensure the DataFrame's index is in datetime format
        data.index = pd.to_datetime(data.index)

        # Select features
        logging.info('Selecting features')
        features = data[config.features_to_include]
        num_features = len(features.columns)  # Get the number of features
        logging.info('Features: ' + str(features.columns.tolist()))  # Log the order of features

        return data

    except Exception as e:
        logging.error(f"Error during data fetching: {e}")
        raise