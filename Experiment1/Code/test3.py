import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

# Function to fetch BTC data and flatten columns
def get_btc_data(ticker='BTC-USD', period='1y', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval)
    data.reset_index(inplace=True)

    # Flatten multi-level column index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)  # Drop ticker index level
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']  # Rename columns

    return data

# Function to plot Close price using seaborn
def plot_ohlc(data):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Close', data=data, label='Close Price')
    plt.title('BTC Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.show()

# Main program
btc_data = get_btc_data()  # Fetch BTC data
print(btc_data.head())  # Print the first 5 rows to confirm structure
plot_ohlc(btc_data)  # Plot Close prices

