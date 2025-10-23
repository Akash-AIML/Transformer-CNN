import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .transformer import TimeSeriesTransformer  # make sure this is correct

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Load Model -----------------
def load_model(input_dim=4, d_model=128, num_heads=8, num_encoder_layers=3, num_decoder_layers=3, output_dim=1):
    model = TimeSeriesTransformer(input_dim=input_dim, d_model=d_model, num_heads=num_heads,
                                  num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                  output_dim=output_dim)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# ----------------- Train if missing -----------------
def train_if_missing(csv_path, seq_len=24, pred_len=12, epochs=50, lr=0.001):
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("Model and scaler found. Loading...")
        model = load_model()
        scaler_mean = np.load(SCALER_PATH)
        scaler_scale = np.load(SCALER_PATH.replace(".npy", "_scale.npy"))
        scaler = StandardScaler()
        scaler.mean_ = scaler_mean
        scaler.scale_ = scaler_scale
        return model, scaler
    
    print("Model or scaler not found. Training new model...")
    df = pd.read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    features = ['Temperature', 'EnergyConsumption', 'HourOfDay', 'DayOfWeek']

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    np.save(SCALER_PATH, scaler.mean_)
    np.save(SCALER_PATH.replace(".npy", "_scale.npy"), scaler.scale_)

    data = df[features].values
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    def create_sequences(dataset):
        X, Y = [], []
        for i in range(len(dataset)-seq_len-pred_len):
            X.append(dataset[i:i+seq_len])
            Y.append(dataset[i+seq_len:i+seq_len+pred_len, 1:2])  # EnergyConsumption
        return np.array(X), np.array(Y)

    X_train, Y_train = create_sequences(train_data)
    X_val, Y_val = create_sequences(val_data)

    model = load_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.to(device)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs = torch.FloatTensor(X_train).to(device)
        targets = torch.FloatTensor(Y_train).to(device)
        outputs = model(inputs, inputs[:, -pred_len:, :])
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model training complete and saved.")
    return model, scaler

# ----------------- Predict -----------------
def predict(input_array, pred_len=12):
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found. You need to train the model first.")
    
    scaler_mean = np.load(SCALER_PATH)
    scaler_scale = np.load(SCALER_PATH.replace(".npy", "_scale.npy"))
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale

    # Convert list of dict to np.array if needed
    if isinstance(input_array, list) and isinstance(input_array[0], dict):
        input_array = np.array([[d['Temperature'], d['EnergyConsumption'], d['HourOfDay'], d['DayOfWeek']] for d in input_array], dtype=np.float32)
    else:
        input_array = np.array(input_array, dtype=np.float32)

    input_scaled = (input_array - scaler.mean_) / scaler.scale_
    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(device)  # [1, seq_len, features]

    model = load_model(input_dim=input_array.shape[1])
    model.eval()

    # Use last seq_len as tgt for autoregressive prediction
    tgt = input_tensor[:, -pred_len:, :]

    with torch.no_grad():
        out = model(input_tensor, tgt)

    # Inverse transform
    dummy = np.zeros((out.shape[1], input_array.shape[1]))
    dummy[:, 1] = out.cpu().numpy()[0, :, 0]
    pred_denorm = dummy[:, 1] * scaler.scale_[1] + scaler.mean_[1]

    return pred_denorm.tolist()

# ----------------- Automatically train/load -----------------
CSV_PATH = "/home/akash/Downloads/synthetic_energy_data.csv"
model, scaler = train_if_missing(CSV_PATH)
