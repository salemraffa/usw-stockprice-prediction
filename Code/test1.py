import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 1. Get Bitcoin data from CoinGecko
def fetch_historical_prices():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '60',  # Nur die letzten 60 Tage historische Daten
        'interval': 'daily'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        prices = [price[1] for price in data['prices']]
        return np.array(prices)
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen der historischen Preise: {e}")
        return None


# 2. Testing data
prices = fetch_historical_prices()
if prices is None or len(prices) < 60:
    print("Nicht genügend Daten für das Modell!")
    exit()

# 3. Daten normalisieren (Skalierung der Preise auf den Bereich [0, 1])
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()


# 4. Funktion zum Erstellen von Sequenzen für das Modell
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


sequence_length = 5  # Die Länge der Eingabesequenzen

# 5. Daten in Trainings-, Validierungs- und Testsets aufteilen
train_prices, temp_prices = train_test_split(prices_scaled, test_size=0.2, shuffle=False)
val_prices, test_prices = train_test_split(temp_prices, test_size=0.5, shuffle=False)

X_train, y_train = create_sequences(train_prices, sequence_length)
X_val, y_val = create_sequences(val_prices, sequence_length)
X_test, y_test = create_sequences(test_prices, sequence_length)

# 6. Umwandeln der Daten in PyTorch-Tensoren
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)


# 7. LSTM Modell definieren
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1])  # Nur die Ausgabe der letzten Zeitschritt verwenden
        return predictions


# 8. Modell initialisieren, Verlustfunktion und Optimierer definieren
model = LSTMModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 9. Training des Modells
epochs = 20
for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation Loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

# 10. Modell evaluieren (auf Testdaten)
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# 11. Vorhersage für den nächsten Tag
model.eval()
with torch.no_grad():
    # Sicherstellen, dass genügend Daten vorhanden sind
    if len(prices_scaled) >= sequence_length:
        last_sequence = torch.tensor(prices_scaled[-sequence_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        predicted_scaled_price = model(last_sequence).item()
        predicted_price = scaler.inverse_transform([[predicted_scaled_price]])[0][0]
        print(f"Predicted price for the next day: ${predicted_price:.2f}")
    else:
        print("Nicht genügend Daten für die Vorhersage!")

# 12. Historische Preise und Vorhersage visualisieren
plt.plot(prices, label="Actual Prices")
plt.axhline(y=predicted_price, color='r', linestyle='--', label="Predicted Price")
plt.legend()
plt.title("Bitcoin Prices and Prediction")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.show()


