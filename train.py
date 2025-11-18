from pathlib import Path
import time

import torch
from torch import nn, optim
from joblib import dump

from dataset import prepare_dataloaders, FEATURE_COLS
from model import LSTMForecaster


# =========================
#   CONFIG
# =========================

SYMBOL = "BTCUSDT"
WINDOW_SIZE = 48
HORIZON = 1
BATCH_SIZE = 64
NUM_EPOCHS = 20

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODELS_DIR / f"lstm_{SYMBOL.lower()}_best.pt"
SCALER_PATH = MODELS_DIR / f"scaler_{SYMBOL.lower()}.joblib"


def get_device():
    if torch.cuda.is_available():
        print("✅ CUDA is available. Using GPU.")
        return torch.device("cuda")
    else:
        print("⚠️ CUDA not available. Using CPU.")
        return torch.device("cpu")


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(1, num_batches)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            running_loss += loss.item()
            num_batches += 1

    return running_loss / max(1, num_batches)


def main():
    device = get_device()

    print(f"Preparing data loaders for symbol: {SYMBOL} ...")
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(
        symbol=SYMBOL,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
    )

    # Save scaler for later use in Streamlit app
    dump(scaler, SCALER_PATH)
    print(f"Feature scaler saved to: {SCALER_PATH}")

    model = LSTMForecaster(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
    ).to(device)

    print(model)
    print()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    print("Starting LSTM training with multivariate features...")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            improved = "✅ (improved, saved)"
        else:
            improved = ""

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} "
            f"- Train Loss: {train_loss:.6f} "
            f"- Val Loss: {val_loss:.6f} {improved}"
        )

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.1f} seconds.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
