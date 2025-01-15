import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from pytrends.request import TrendReq

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# Function to fetch BTC data and flatten columns
def get_btc_data(ticker='BTC-USD', period='1y', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval)
    data.reset_index(inplace=True)

    # Rename columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    return data


# Function to fetch Google Trends data
def get_google_trends_data(keyword='bitcoin', timeframe='2023-01-01 2024-01-01'):
    pytrends = TrendReq()
    pytrends.build_payload(kw_list=[keyword], timeframe=timeframe)
    trends_data = pytrends.interest_over_time()
    trends_data.reset_index(inplace=True)
    trends_data = trends_data[['date', keyword]]
    trends_data.columns = ['Date', 'Interest']
    return trends_data


# Function to filter data by timeframe
def filter_by_timeframe(data, timeframe='daily'):
    data['Date'] = pd.to_datetime(data['Date'])
    if timeframe == 'weekly':
        return data.resample('W', on='Date').agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
             'Interest': 'mean'}).reset_index()
    elif timeframe == 'monthly':
        return data.resample('M', on='Date').agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
             'Interest': 'mean'}).reset_index()
    else:
        return data


# Simple Moving Average (SMA)
def add_sma(data, window=7):
    data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
    return data


# Relative Strength Index (RSI)
def add_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data


# Plot function for BTC prices
def plot_ohlc(data):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Close', data=data, label='Close Price')
    plt.title('BTC Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.show()


# Create sequences for LSTM
def create_sequences(data, seq_length=50):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]  # Include both Close price and Interest
        label = data[i + seq_length][0]  # Label is still the Close price
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


# LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get last time step
        return out


# Main Program
btc_data = get_btc_data()  # Fetch BTC data
trends_data = get_google_trends_data('bitcoin')  # Fetch Google Trends data
print(trends_data.head())  # Check first 5 rows of trends data

# Merge BTC data and Google Trends data
btc_data = pd.merge(btc_data, trends_data, on='Date', how='outer').fillna(0)  # Fill missing Interest values with 0
print(btc_data.head())  # Confirm Interest column is added

# Plot BTC Close Price vs Google Trends Interest
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Close', data=btc_data, label='BTC Close Price')
sns.lineplot(x='Date', y='Interest', data=btc_data, label='Google Search Interest', color='orange')
plt.title('BTC Price vs Google Search Interest')
plt.show()

# Filter data for weekly view
filtered_btc_data = filter_by_timeframe(btc_data, timeframe='weekly')
plot_ohlc(filtered_btc_data)  # Plot the filtered data (e.g., weekly prices)

# Add SMA and RSI
btc_data = add_sma(btc_data, window=7)
btc_data = add_rsi(btc_data, window=14)

# Normalize data
scaler = MinMaxScaler()
btc_data[['Close', 'Interest']] = scaler.fit_transform(btc_data[['Close', 'Interest']])

# Create LSTM sequences
seq_length = 50
sequences, labels = create_sequences(btc_data[['Close', 'Interest']].values, seq_length=seq_length)

# Initialize and train the LSTM model
model = LSTMModel(input_size=2, hidden_size=64, num_layers=2)  # input_size=2 for Close and Interest
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    predictions = model(sequences)
    loss = criterion(predictions, labels.unsqueeze(-1))

    # Backward pass
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}")
