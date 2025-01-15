# usw-stock-price-prediction

Salem Raffa


Project Goals and Objectives:

Goal: To develop an AI-based stock price prediction model that forecasts the percentage change in price for the next day.

Key Objectives:

Data Collection: Acquire financial market data using APIs like yfinance or datasets from platforms like Kaggle.
Data Understanding and Cleaning: Explore the dataset to understand the structure, remove anomalies, and fill or drop missing values.
Feature Engineering: Create features such as moving averages, Relative Strength Index (RSI), and price correlations to improve the model.
Model Development: Build and train a multi-input LSTM neural network to predict price changes.
Performance Evaluation: Use metrics like mean squared error (MSE) to evaluate the model's accuracy.
Experimentation: Test different configurations (e.g., attention mechanisms, different numbers of LSTM layers) to optimize the model.


Problem Statement:
Financial markets are complex and influenced by a variety of factors. Predicting future price movements based solely on historical trends is challenging. Your project aims to use advanced deep learning techniques to capture these patterns and make informed predictions to assess the potential price increase of an asset (e.g., Bitcoin or stocks).

Tools and Technologies:
Data Fetching: yfinance, APIs like CryptoDataDownload.
Data Visualization: seaborn, matplotlib.
Preprocessing and Feature Engineering: pandas, talib (for indicators).
Deep Learning Framework: TensorFlow/Keras for building LSTM models.
Evaluation: Scikit-learn for performance metrics (e.g., mean_squared_error, classification_report).
Expected Outcome:
A trained and tested LSTM model that:

Predicts the percentage increase or decrease in stock/crypto prices for the next time period.
Outperforms baseline dummy strategies (e.g., random buy/sell) and compares to market benchmarks.
Provides visual insights through plots of predictions vs. actual prices.
