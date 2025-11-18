from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from data_prep import load_symbol_df, train_val_test_split, DEFAULT_SYMBOL


# =========================
#   CONFIG
# =========================

# Feature columns we'll feed into the model
FEATURE_COLS: List[str] = [
    "log_return",
    "rolling_mean_24",
    "rolling_std_24",
    "volume_norm",
]

# What we are trying to predict (still log_return)
TARGET_COL = "log_return"


# =========================
#   FEATURE ENGINEERING
# =========================

def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from the 'close' column and add as 'log_return'.

    r_t = log(close_t / close_{t-1})

    The first row has no previous price -> NaN, we keep it for now.
    """
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all features used in the multivariate model.

    - log_return:            log(close_t / close_{t-1})
    - rolling_mean_24:       24-step rolling mean of log_return
    - rolling_std_24:        24-step rolling std (volatility proxy)
    - volume_norm:           volume_from normalized by its 24-step rolling mean

    We drop any rows that have NaNs in these feature columns
    (caused by the rolling operations and first log_return).
    """
    df = add_log_return(df)

    df["rolling_mean_24"] = df["log_return"].rolling(window=24).mean()
    df["rolling_std_24"] = df["log_return"].rolling(window=24).std()

    vol_roll_mean = df["volume_from"].rolling(window=24).mean()
    df["volume_norm"] = (df["volume_from"] / vol_roll_mean) - 1.0

    # Drop rows where any of our feature columns are NaN
    df = df.dropna(subset=FEATURE_COLS)

    return df


# =========================
#   SCALING (FEATURES ONLY)
# =========================

def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale feature columns using StandardScaler. TARGET (log_return) is kept
    in its original scale (already small values).

    Returns:
    - X_train_scaled, X_val_scaled, X_test_scaled: arrays of shape (n, num_features)
    - scaler: fitted StandardScaler
    """
    scaler = StandardScaler()

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# =========================
#   SLIDING WINDOW CREATION
# =========================

def create_sliding_windows(
    features: np.ndarray,
    targets: np.ndarray,
    window_size: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input/output pairs from multivariate features and 1D targets.

    features: (N, num_features)
    targets:  (N,)  - log_return values (unscaled)

    For each i, we create:

      X[i] = features[i : i+window_size, :]          -> (window_size, num_features)
      y[i] = targets[i+window_size : i+window_size+horizon]  -> (horizon,)

    Returns:
    - X: (num_samples, window_size, num_features)
    - y: (num_samples, horizon)
    """
    if features.ndim != 2:
        raise ValueError("Expected 'features' to be 2D (N, num_features).")
    if targets.ndim != 1:
        raise ValueError("Expected 'targets' to be 1D (N,).")

    n = len(targets)
    num_features = features.shape[1]

    num_samples = n - window_size - horizon + 1
    if num_samples <= 0:
        raise ValueError("Not enough data for the given window_size and horizon.")

    X = np.zeros((num_samples, window_size, num_features), dtype=np.float32)
    y = np.zeros((num_samples, horizon), dtype=np.float32)

    for i in range(num_samples):
        X[i] = features[i : i + window_size, :]
        y[i] = targets[i + window_size : i + window_size + horizon]

    return X, y


# =========================
#   PYTORCH DATASET
# =========================

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series windows.

    - X: (num_samples, seq_len, num_features)
    - y: (num_samples, horizon)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
#   HIGH-LEVEL PREP FUNCTION
# =========================

def prepare_dataloaders(
    symbol: str = DEFAULT_SYMBOL,
    window_size: int = 48,
    horizon: int = 1,
    batch_size: int = 64,
):
    """
    End-to-end:

    1. Load OHLCV for a symbol.
    2. Add engineered features (log_return, rolling stats, volume_norm).
    3. Split into train/val/test (chronological).
    4. Scale features with StandardScaler (fit on train only).
    5. Build sliding windows for each split.
    6. Wrap into PyTorch DataLoaders.

    Returns:
    - train_loader, val_loader, test_loader, scaler
    """

    # 1) Load full dataframe
    df = load_symbol_df(symbol)

    # 2) Feature engineering
    df = add_features(df)

    # 3) Chronological split
    train_df, val_df, test_df = train_val_test_split(df)

    # 4) Scale features (NOT targets)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        train_df, val_df, test_df, feature_cols=FEATURE_COLS
    )

    # Targets in original log-return scale
    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # 5) Sliding windows
    X_train, y_train_win = create_sliding_windows(X_train_scaled, y_train, window_size, horizon)
    X_val, y_val_win = create_sliding_windows(X_val_scaled, y_val, window_size, horizon)
    X_test, y_test_win = create_sliding_windows(X_test_scaled, y_test, window_size, horizon)

    # 6) Datasets & loaders
    train_dataset = TimeSeriesDataset(X_train, y_train_win)
    val_dataset = TimeSeriesDataset(X_val, y_val_win)
    test_dataset = TimeSeriesDataset(X_test, y_test_win)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
