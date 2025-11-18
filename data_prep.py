from pathlib import Path

import numpy as np
import pandas as pd


# === CONFIGURATION ===
# Default symbol: we start with BTC as you requested.
DEFAULT_SYMBOL = "BTCUSDT"
DATA_DIR = Path(__file__).parent / "data"


def load_symbol_df(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """
    Load raw OHLCV time series for a given symbol (e.g. 'BTCUSDT').

    Returns a pandas DataFrame with:
    - A proper datetime index.
    - Columns: ['open', 'high', 'low', 'close', 'volume_from', 'volume_to'] as float32.
    """

    file_path = DATA_DIR / f"{symbol}_1h.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    # Read CSV using pandas
    df = pd.read_csv(file_path)

    # --- timestamp parsing ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by time just to be safe
    df = df.sort_values("timestamp")

    # Set timestamp as index (common pattern in time series)
    df = df.set_index("timestamp")

    # Ensure numeric columns are float32 (less memory, enough precision)
    numeric_cols = ["open", "high", "low", "close", "volume_from", "volume_to"]
    df[numeric_cols] = df[numeric_cols].astype("float32")

    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
    """
    Chronological split of the time series into train, validation, and test sets.

    - train_frac: fraction of data for training (e.g. 0.7 = 70% oldest data).
    - val_frac: fraction of remaining for validation (e.g. 0.15 = 15%).
      The rest goes to the test set.

    IMPORTANT: we do NOT shuffle, because order matters in time series.
    """

    n = len(df)
    if n == 0:
        raise ValueError("DataFrame is empty, cannot split.")

    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def main():
    """
    Small sanity check: load one symbol and print some info.
    This runs only when you execute this file directly:

        python data_prep.py
    """
    symbol = DEFAULT_SYMBOL
    df = load_symbol_df(symbol)

    print(f"Loaded {symbol} data:")
    print(df.head())
    print()
    print(f"Number of rows: {len(df)}")
    print(f"Time range: {df.index.min()}  ->  {df.index.max()}")

    train_df, val_df, test_df = train_val_test_split(df)

    print()
    print("Split sizes:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Test:  {len(test_df)} rows")


if __name__ == "__main__":
    main()
