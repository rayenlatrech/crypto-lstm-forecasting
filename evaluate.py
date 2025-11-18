from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import load

from data_prep import load_symbol_df, train_val_test_split
from dataset import add_features, scale_features, create_sliding_windows, TimeSeriesDataset, FEATURE_COLS, TARGET_COL
from model import LSTMForecaster


SYMBOL = "BTCUSDT"
WINDOW_SIZE = 48
HORIZON = 1
BATCH_SIZE = 64

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

MODELS_DIR = Path(__file__).parent / "models"
BEST_MODEL_PATH = MODELS_DIR / f"lstm_{SYMBOL.lower()}_best.pt"
SCALER_PATH = MODELS_DIR / f"scaler_{SYMBOL.lower()}.joblib"


def get_device():
    if torch.cuda.is_available():
        print("✅ CUDA is available. Using GPU.")
        return torch.device("cuda")
    else:
        print("⚠️ CUDA not available. Using CPU.")
        return torch.device("cpu")


def main():
    device = get_device()

    print(f"Loading data for {SYMBOL} ...")
    df = load_symbol_df(SYMBOL)
    df = add_features(df)
    train_df, val_df, test_df = train_val_test_split(df)

    print("Scaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        train_df, val_df, test_df, feature_cols=FEATURE_COLS
    )

    # Targets (log returns) in original scale
    y_test = test_df[TARGET_COL].values

    print("Creating sliding windows for TEST set...")
    X_test, y_test_win = create_sliding_windows(
        X_test_scaled,
        y_test,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
    )

    test_dataset = TimeSeriesDataset(X_test, y_test_win)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loading best LSTM model from: {BEST_MODEL_PATH}")
    model = LSTMForecaster(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
    ).to(device)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).reshape(-1)
    targets = np.concatenate(all_targets, axis=0).reshape(-1)

    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)

    print("\n=== Test metrics on log-return scale (multivariate features) ===")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MSE:  {mse:.6f}")

    N_PLOT = 500
    if len(preds) < N_PLOT:
        N_PLOT = len(preds)

    plt.figure(figsize=(12, 6))
    plt.plot(targets[-N_PLOT:], label="True log return")
    plt.plot(preds[-N_PLOT:], label="Predicted log return")
    plt.xlabel("Time steps (test window)")
    plt.ylabel("log return")
    plt.title(f"{SYMBOL} - True vs Predicted log returns (Test, multivariate)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
