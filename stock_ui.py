import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

MODEL_PATH = "C:/Users/user/OneDrive/Desktop/CODE/Stock-Market-Predictor/stock_model.h5"
model = load_model(MODEL_PATH, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

api_key = "B93Z6SS830BN9LRW"

def fetch_stock_data(symbol, api_key, output_size="compact"):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": output_size,
        "datatype": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.rename(columns={"4. close": "Close"})
        df.index = pd.to_datetime(df.index)
        df = df[["Close"]].astype(float).sort_index()
        return df
    else:
        st.error(f"Error fetching data: {data}")
        return None

def predict_next_days(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1,1))

    seq_length = 60  
    last_60_days = scaled_data[-seq_length:]
    X_input = np.array([last_60_days])
    X_input = X_input.reshape((1, seq_length, 1))

    predictions = []
    for _ in range(7):
        predicted_price = model.predict(X_input)[0][0]
        predictions.append(predicted_price)
        X_input = np.append(X_input[:, 1:, :], [[[predicted_price]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7)
    prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Close": predictions.flatten()})
    return prediction_df

def plot_stock_data(df, symbol, predictions):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["Close"], label=f"{symbol} Closing Price", color='blue', linewidth=2)
    plt.plot(predictions["Date"], predictions["Predicted Close"], label="Predicted Prices", linestyle='dashed', color='red', marker='o')
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.title(f"{symbol} Stock Price & LSTM Prediction")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

st.title("📈 LSTM Stock Price Predictor")
st.write("Enter a stock ticker to fetch historical data and predict future prices.")

symbol = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, GOOGL):", "").upper()

if st.button("Fetch & Predict"):
    if not symbol:
        st.warning("Please enter a stock ticker symbol.")
    else:
        stock_data = fetch_stock_data(symbol, api_key)
        if stock_data is not None:
            predictions = predict_next_days(stock_data)

            plot_stock_data(stock_data, symbol, predictions)
            st.subheader(f"Stock Data for {symbol}")
            st.dataframe(stock_data.tail(10))

            st.subheader("Predicted Prices for Next 7 Days")
            st.dataframe(predictions)