# Stock Price Predictor

This repository contains a machine learning project that predicts stock prices. It uses the H2O AutoML library to automatically train and tune a variety of machine learning models.

## Scripts

- `config.py`: Contains the configuration settings for the project.
- `model_training.py`: Responsible for training the machine learning model.
- `main.py`: The main script that runs the entire project.
- `db_operations.py`: Contains functions for interacting with a PostgreSQL database.
- `model_evaluation.py`: Contains functions for evaluating the performance of the machine learning model.
- `data_fetching.py`: Fetches and preprocesses the stock price data.

## How to Run

1. Set your configuration settings in `config.py`.
2. Run `main.py` to fetch the data, train the model, evaluate its performance, and save the results to the database.

## Dependencies

- H2O
- pandas
- numpy
- pandas_datareader
- yfinance
- sklearn
- psycopg2

## License

This project is licensed under the terms of the MIT license.
