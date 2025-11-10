# Stock-Predictor-using-Streamlit

Stock Predictor with Polynomial Regression

This is an interactive stock prediction web app built using Streamlit that allows users to explore historical stock prices and generate polynomial regression-based forecasts. The app is purely mathematical and not financial advice—designed for educational purposes and learning regression concepts with real market data.

Features

Search any stock ticker (e.g., AAPL, MSFT, TSLA, RELIANCE.NS).

Interactive Plotly chart showing:

Historical stock prices

Fitted polynomial regression line

Forecasted prices for the next N years (dotted/dashed line)

Polynomial regression customization:

Adjust regression degree using a slider

View polynomial coefficients

Matplotlib integration: static chart of historical + predicted prices.

Data preview: tail of historical data + head of predicted values.

Dark/Light theme support (optional custom toggle).

Resilient to missing data: input validation and error handling for invalid tickers or insufficient data.

Tech Stack

Python 3.10+

Streamlit: Web app interface

YFinance: Fetch historical stock data

NumPy & Pandas: Data manipulation and regression

Plotly: Interactive plots

Matplotlib: Static plots

How It Works

User enters a stock ticker and selects a start/end date.

App fetches historical adjusted close prices using yfinance.

Polynomial regression is applied using numpy.polyfit and numpy.poly1d.

A fitted line is generated over historical prices, and forecasted points are calculated for the next N years.

An interactive Plotly chart visualises historical, fitted, and forecasted prices.

Polynomial coefficients and a data preview are displayed for further insight.

Key Features Explained

Dotted/Dashed Lines: The fitted line uses dash='dot' for historical regression; forecast uses dash='dash'.

Polynomial Degree: Controls the flexibility of the regression. Higher degrees can fit past data more closely but may overfit.

Coefficients: Represent the terms of the polynomial equation; highest degree first.

Theme: Light/dark background can be customised using Streamlit or CSS.
