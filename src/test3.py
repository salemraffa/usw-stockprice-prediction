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

    # Flatten multi-level column index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)  # Drop ticker index level
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']  # Rename columns

    return data


# Function to fetch Google Trends data for a specific keyword
def get_google_trends_data(keyword='bitcoin', timeframe='today 12-m'):
    pytrends = TrendReq()
    pytrends.build_payload(kw_list=[keyword], timeframe=timeframe)
    trends_data = pytrends.interest_over_time()  # Get interest over time
    trends_data.reset_index(inplace=True)
    trends_data = trends_data[['date', keyword]]  # Keep only date and interest columns
    trends_data.columns = ['Date', 'Interest']  # Rename columns for clarity
    return trends_data


# Function to filter data by timeframe
def filter_by_timeframe(data, timeframe='daily'):
    data['Date'] = pd.to_datetime(data['Date'])  # Convert date column to datetime if not already

    if timeframe == 'weekly':
        return data.resample('W', on='Date').agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).reset_index()
    elif timeframe == 'monthly':
        return data.resample('M', on='Date').agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).reset_index()
    else:
        return data  # Return daily data if no timeframe is selected


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


# Function to plot Close price using seaborn
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
        seq = data[i:i + seq_length, :-1]  # Include both Close price and Interest
        label = data[i + seq_length, 0]  # Label is still the Close price
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


# LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Fully connected layer for final output

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get last time step output
        return out


# Main program
btc_data = get_btc_data()  # Fetch BTC data


# Fetch Google Trends data
trends_data = get_google_trends_data('bitcoin', timeframe='today 12-m')
print(trends_data.head())  # Check first 5 rows of trends data

# Merge BTC data and Google Trends data
btc_data = pd.merge(btc_data, trends_data, on='Date', how='inner')  # Merge on Date
print(btc_data.head())  # Confirm Interest column is added


plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Close', data=btc_data, label='BTC Close Price')
sns.lineplot(x='Date', y='Interest', data=btc_data, label='Google Search Interest', color='orange')
plt.title('BTC Price vs Google Search Interest')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

print(btc_data.head())  # Print the first 5 rows to confirm structure

# Apply filtering for the presentation:
filtered_btc_data = filter_by_timeframe(btc_data, timeframe='weekly')  # Options: 'daily', 'weekly', 'monthly'
plot_ohlc(filtered_btc_data)  # Plot the filtered data (e.g., weekly prices)

btc_data = add_sma(btc_data, window=7)  # Add 7-day Simple Moving Average
btc_data = add_rsi(btc_data, window=14)  # Add 14-day RSI

# Normalize the data
scaler = MinMaxScaler()
btc_data['Close'] = scaler.fit_transform(btc_data[['Close']])  # Normalize Close prices

# Create LSTM input sequences (with Close and Interest as features)
seq_length = 50
sequences, labels = create_sequences(btc_data[['Close', 'Interest']].values, seq_length=seq_length)


# Initialize and train the model
model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    predictions = model(sequences.unsqueeze(-1))
    loss = criterion(predictions, labels.unsqueeze(-1))

    # Backward pass
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
