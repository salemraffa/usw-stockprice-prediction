import requests
import pandas as pd
import matplotlib.pyplot as plt


# Function to fetch BTC price data

# This function fetches the Bitcoin price data for the last 30 days using CoinGecko's API and converts it into a pandas DataFrame.
def fetch_btc_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        prices = data['prices']  # list of [timestamp, price]

        # Convert to DataFrame
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to readable format
        return df
    else:
        print("Error fetching data")
        return None

def plot_btc_price(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['price'], label='BTC Price (USD)', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Bitcoin Price Over Last 30 Days')
    plt.legend()
    plt.show()

def add_sma(df, window=5):
    df[f'SMA_{window}'] = df['price'].rolling(window=window).mean()  # Calculate moving average
    return df

# Fetch data and run the analysis
btc_data = fetch_btc_data()

if btc_data is not None:
    btc_data = add_sma(btc_data, window=7)  # 7-day SMA
    plot_btc_price(btc_data)

from sklearn.linear_model import LinearRegression
import numpy as np


def predict_btc_price(df):
    # Prepare the data for regression
    df['timestamp_num'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    X = df[['timestamp_num']]
    y = df['price']

    model = LinearRegression()
    model.fit(X, y)

    # Predict the next day
    future_timestamp = df['timestamp_num'].max() + 86400  # 1 day later (in seconds)
    future_timestamp_df = pd.DataFrame({'timestamp_num': [future_timestamp]})  # Create DataFrame with column name

    future_price = model.predict(future_timestamp_df)[0]  # Pass DataFrame instead of list

    print(f"Predicted BTC price for tomorrow: ${future_price:.2f}")


predict_btc_price(btc_data)
