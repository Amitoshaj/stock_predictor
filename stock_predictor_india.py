import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

st.title("📈 Indian Stock Price Predictor")

# Select stock
ticker = st.text_input("Enter NSE stock ticker (e.g., TCS.NS, INFY.NS)", "TCS.NS")

# Select date range
start_date = st.date_input("Start Date", datetime(2015, 1, 1))
end_date = st.date_input("End Date", datetime.today())

# Load data
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

st.subheader("Raw Stock Data")
st.write(data.tail())

# Plot closing price
st.subheader("Closing Price Trend")
fig, ax = plt.subplots()
ax.plot(data['Date'], data['Close'])
plt.xticks(rotation=45)
st.pyplot(fig)

# Linear Regression for basic prediction
st.subheader("Price Prediction")

# Use closing price for model
data['Days'] = (data['Date'] - data['Date'].min()).dt.days
X = np.array(data['Days']).reshape(-1, 1)
y = data['Close'].values

model = LinearRegression()
model.fit(X, y)

# Predict next 30 days
future_days = 30
future_X = np.array([X[-1][0] + i for i in range(1, future_days + 1)]).reshape(-1, 1)
future_dates = [data['Date'].max() + timedelta(days=i) for i in range(1, future_days + 1)]
future_preds = model.predict(future_X)

# Plot predictions
st.subheader(f"Next {future_days} Days Prediction")
fig2, ax2 = plt.subplots()
ax2.plot(data['Date'], data['Close'], label="Historical")
ax2.plot(future_dates, future_preds, label="Predicted", linestyle="--")
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig2)
