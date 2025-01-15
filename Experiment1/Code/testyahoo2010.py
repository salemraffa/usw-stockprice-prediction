import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Fetch Bitcoin data from Yahoo Finance
def fetch_historical_prices():
    btc = yf.Ticker("BTC-USD")
    historical_data = btc.history(period="max")  # Get all available data
    close_prices = historical_data['Close'].values  # Extract only the closing prices
    return close_prices

# 2. Fetch data
prices = fetch_historical_prices()
if prices is None or len(prices) < 10:  # Check if at least 10 data points exist
    print("Not enough data for the model!")
    exit()

# 3. Normalize data (scale prices to [0, 1] range)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

# 4. Create sequences for the model
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 50  # Increased sequence length for better pattern recognition

# 5. Split data into training, validation, and test sets
train_prices, temp_prices = train_test_split(prices_scaled, test_size=0.2, shuffle=False)  # 80% train, 20% validation + test
val_prices, test_prices = train_test_split(temp_prices, test_size=0.5, shuffle=False)  # 10% validation, 10% test

X_train, y_train = create_sequences(train_prices, sequence_length)
X_val, y_val = create_sequences(val_prices, sequence_length)
X_test, y_test = create_sequences(test_prices, sequence_length)

# 6. Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

print(f"Data Points - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# 7. Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=256, output_size=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=3, batch_first=True)  # Added 3 LSTM layers
        self.dropout = nn.Dropout(dropout_rate)  # Add dropout to prevent overfitting
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1])  # Apply dropout to the last time step's output
        predictions = self.linear(lstm_out)
        return predictions

# 8. Initialize model, loss function, and optimizer
model = LSTMModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Lower learning rate and L2 regularization

# 9. Train the model with early stopping
epochs = 100  # Increased epochs for better convergence
patience = 10  # Early stopping patience increased
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Calculate validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0  # Reset early stopping counter if validation loss improves
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# 10. Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# 11. Predict the next day's price
model.eval()
with torch.no_grad():
    if len(prices_scaled) >= sequence_length:
        last_sequence = torch.tensor(prices_scaled[-sequence_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        predicted_scaled_price = model(last_sequence).item()
        predicted_price = scaler.inverse_transform([[predicted_scaled_price]])[0][0]
        print(f"Predicted price for the next day: ${predicted_price:.2f}")
    else:
        print("Not enough data for prediction!")

# 12. Plot actual prices and prediction
plt.plot(prices, label="Actual Prices")
plt.axhline(y=predicted_price, color='r', linestyle='--', label="Predicted Price")
plt.legend()
plt.title("Bitcoin Prices and Prediction")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.show()
