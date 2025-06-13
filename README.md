# Indian Stock Price Predictor 📈

A Streamlit app that fetches and predicts stock prices for Indian companies (NSE) using yfinance and Linear Regression.

## Features
- Fetches historical stock data (e.g., TCS.NS, RELIANCE.NS)
- Visualizes historical prices
- Predicts next 30 days using Linear Regression
- Built with Streamlit

## How to Run

```bash
pip install streamlit yfinance scikit-learn pandas matplotlib
streamlit run stock_predictor_india.py
