import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Predictor with Polynomial Forecast")
st.write(
    "Enter a ticker symbol to see historical prices and a polynomial regression forecast. "
    "_Purely mathematical — not financial advice._"
)

with st.sidebar:
    st.header("Settings")
    ticker_input = st.text_input("Ticker", "AAPL").strip().upper()
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*5))
    end_date = st.date_input("End Date", datetime.now())
    prediction_years = st.slider("Forecast Horizon (Years)", 1, 10, 5)
    degree = st.slider("Polynomial Degree", 1, 5, 2)

if not ticker_input:
    st.info("Enter a ticker to begin.")
    st.stop()
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False, auto_adjust=True)
        if df.empty or 'Close' not in df.columns:
            return None
        return df[['Close']].rename(columns={'Close': 'Price'})
    except:
        return None

with st.spinner(f"Fetching data for {ticker_input}..."):
    df = fetch_data(ticker_input, start_date, end_date)

if df is None or len(df) == 0:
    st.error(
        f"No data found for `{ticker_input}`.\n"
        "Check the ticker, date range, or your internet connection."
    )
    st.stop()
if len(df) < degree + 1:
    st.error(
        f"Not enough data points ({len(df)}) for polynomial degree {degree}.\n"
        "Increase the date range or reduce the degree."
    )
    st.stop()

x = np.array([d.toordinal() for d in df.index])
y = df['Price'].to_numpy().flatten()

try:
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
except Exception as e:
    st.error(f"Regression failed: {e}")
    st.stop()

fitted = poly(x)
last_date = df.index[-1]
future_dates = pd.date_range(
    start=last_date + timedelta(days=1),
    end=last_date + pd.DateOffset(years=prediction_years),
    freq='B'
)
x_future = np.array([d.toordinal() for d in future_dates])
y_future = poly(x_future)

historical = pd.DataFrame({'Date': df.index, 'Price': y, 'Fitted': fitted}).set_index('Date')
forecast = pd.DataFrame({'Date': future_dates, 'Predicted': y_future}).set_index('Date')

fig = go.Figure()
fig.add_trace(go.Scatter(x=historical.index, y=historical['Price'], mode='lines', name='Historical'))
fig.add_trace(go.Scatter(x=historical.index, y=historical['Fitted'], mode='lines', name=f'Fitted (deg {degree})', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Predicted'], mode='lines', name=f'Forecast ({prediction_years}y)', line=dict(dash='dash')))
fig.update_layout(
    title=f"{ticker_input} – Price + Polynomial Forecast",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode='x unified',
    template='plotly_white',
    height=600
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Data Preview")
preview = pd.concat([
    historical.tail(10).reset_index(),
    pd.DataFrame({"Date": ["..."], "Price": ["..."], "Fitted": ["..."]}),
    forecast.head(10).reset_index()
], ignore_index=True)
st.dataframe(preview)

st.subheader("Polynomial Coefficients")
coeff_df = pd.DataFrame({
    "Term": [f"x^{i}" for i in range(degree, -1, -1)],
    "Coefficient": [f"{c:.2e}" for c in coeffs]
})
st.table(coeff_df)

st.caption("⚠️ This is purely mathematical. Not financial advice. Polynomial regression cannot reliably predict stock prices.")
