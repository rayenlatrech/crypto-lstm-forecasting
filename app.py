import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from joblib import load
from pathlib import Path
import matplotlib.pyplot as plt

from data_prep import load_symbol_df
from dataset import add_features, FEATURE_COLS
from model import LSTMForecaster


# Must match training config
SYMBOL = "BTCUSDT"
WINDOW_SIZE = 48
HORIZON = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

MODELS_DIR = Path(__file__).parent / "models"
BEST_MODEL_PATH = MODELS_DIR / f"lstm_{SYMBOL.lower()}_best.pt"
SCALER_PATH = MODELS_DIR / f"scaler_{SYMBOL.lower()}.joblib"


@st.cache_resource
def load_model_and_scaler():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = load(SCALER_PATH)

    model = LSTMForecaster(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
    ).to(device)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    return model, scaler, device


@st.cache_data
def load_prepared_data():
    df = load_symbol_df(SYMBOL)
    df = add_features(df)
    return df


def main():
    st.title("BTCUSDT LSTM Forecaster (log returns, multivariate)")

    st.write(
        """
        This demo uses an LSTM trained on BTCUSDT hourly data with engineered features:
        log returns, rolling mean/std of returns, and normalized volume.
        It predicts the next log return and converts it to a one-step-ahead price.
        """
    )

    df = load_prepared_data()
    model, scaler, device = load_model_and_scaler()

    st.sidebar.header("Settings")
    n_history = st.sidebar.slider("Plot history window (hours)", 100, 2000, 500)

    # Use last WINDOW_SIZE rows as input window
    if len(df) < WINDOW_SIZE + 1:
        st.error("Not enough data to build a context window.")
        return

    recent_df = df.iloc[-n_history:]
    st.subheader("Recent BTCUSDT close prices")
    st.line_chart(recent_df["close"])

    # Build model input from last WINDOW_SIZE feature rows
    context_df = df.iloc[-WINDOW_SIZE:]
    features = context_df[FEATURE_COLS].values  # (WINDOW_SIZE, num_features)
    features_scaled = scaler.transform(features)

    X_input = torch.from_numpy(features_scaled.astype(np.float32)).unsqueeze(0)  # (1, seq_len, num_features)
    X_input = X_input.to(device)

    with torch.no_grad():
        pred_log_return = model(X_input).cpu().numpy().flatten()[0]

    last_close = context_df["close"].iloc[-1]
    predicted_close = float(last_close * np.exp(pred_log_return))

    st.subheader("Next-step forecast")
    st.write(f"Last close price: **{last_close:,.2f} USDT**")
    st.write(f"Predicted log return: **{pred_log_return:.5f}**")
    st.write(f"Predicted next close: **{predicted_close:,.2f} USDT**")

    # Plot last history plus predicted point
    st.subheader("Price with one-step-ahead prediction")

    hist_prices = recent_df["close"].copy()
    fut_index = hist_prices.index[-1] + (hist_prices.index[-1] - hist_prices.index[-2])

    future_series = pd.Series(
        [predicted_close],
        index=[fut_index],
        name="Predicted close"
    )

    combined = pd.concat([hist_prices, future_series])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_prices.index, hist_prices.values, label="Historical close")
    ax.scatter(future_series.index, future_series.values, label="Predicted next close")
    ax.set_xlabel("Time")
    ax.set_ylabel("BTCUSDT close price")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)


if __name__ == "__main__":
    main()
