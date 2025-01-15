sequence_length = 10  # Increased from 5 to 10 days

# Recreate the sequences for training, validation, and testing
X_train, y_train = create_sequences(train_prices, sequence_length)
X_val, y_val = create_sequences(val_prices, sequence_length)
X_test, y_test = create_sequences(test_prices, sequence_length)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
