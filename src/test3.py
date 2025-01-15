import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from pytrends.request import TrendReq

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler




# Function to plot Close price using seaborn
def plot_ohlc(data):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Close', data=data, label='Close Price')
    plt.title('BTC Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.show()

