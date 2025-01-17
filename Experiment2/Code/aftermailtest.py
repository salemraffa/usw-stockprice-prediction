import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Fetch Bitcoin data from Yahoo Finance
def fetch_historical_prices():
    btc = yf.Ticker("BTC-USD")
    historical_data = btc.history(start="2020-01-01")  # Verwenden Sie Daten ab 2020
    close_prices = historical_data['Close'].values  # Extrahieren Sie die Schlusskurse
    return close_prices

# 2. Fetch data
prices = fetch_historical_prices()
if prices is None or len(prices) < 10:
    print("Nicht genügend Daten für das Modell!")
    exit()

# 3. Berechnung der prozentualen Preissteigerung
price_changes = (prices[1:] - prices[:-1]) / prices[:-1]  # Prozentuale Änderung berechnen
prices = prices[1:]  # Entfernen Sie den ersten Wert, da keine Änderung berechnet werden kann

# 4. Sequenzen erstellen
def create_sequences(data, targets, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(targets[i + sequence_length - 1])
    return np.array(X), np.array(y)

sequence_length = 50  # Verwenden Sie eine Sequenzlänge von 50 Tagen
X_data, y_data = create_sequences(prices, price_changes, sequence_length)

# 5. Daten in Trainings-, Validierungs- und Testsets aufteilen
X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.2, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# 6. In PyTorch-Tensoren konvertieren
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# 7. LSTM-Modell definieren
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=256, output_size=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=3, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1])
        predictions = self.linear(lstm_out)
        return predictions

# 8. Modell initialisieren, Verlustfunktion und Optimierer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

# 9. Modell trainieren
epochs = 100
patience = 10
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# 10. Modell testen
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# 11. Vorhersage berechnen
model.eval()
with torch.no_grad():
    last_sequence = torch.tensor(prices[-sequence_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    predicted_change = model(last_sequence).item()
    predicted_price = prices[-1] * (1 + predicted_change)
    print(f"Vorhergesagter Preis für den nächsten Tag: ${predicted_price:.2f}")
