import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """
    LSTM-based model for time series forecasting.

    Input shape per batch:
        (batch_size, seq_len, input_size)

    Output shape per batch:
        (batch_size, horizon)

    Example:
        model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, dropout=0.2, horizon=1)
        y_pred = model(X_batch)  # X_batch: (B, seq_len, 1)
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        # LSTM layer(s).
        # batch_first=True means input/output tensors are shaped as (batch, seq, feature)
        # instead of (seq, batch, feature). This matches our Dataset design.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # dropout only between layers
        )

        # Fully connected (linear) layer to map the last hidden state to the desired horizon.
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_size)

        Returns:
        - out: (batch_size, horizon)
        """
        # LSTM output:
        #   lstm_out: (batch_size, seq_len, hidden_size)
        #   (h_n, c_n): hidden and cell states for all layers, final time step
        lstm_out, (h_n, c_n) = self.lstm(x)

        # We take the hidden state at the LAST time step for each sequence.
        # lstm_out[:, -1, :] has shape: (batch_size, hidden_size)
        last_hidden = lstm_out[:, -1, :]

        # Map hidden state to horizon steps
        out = self.fc(last_hidden)  # (batch_size, horizon)

        return out

class GRUForecaster(nn.Module):
    """
    GRU-based model for time series forecasting.

    Same interface as LSTMForecaster, but uses GRU cells instead of LSTM.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_size)
        returns: (batch_size, horizon)
        """
        gru_out, h_n = self.gru(x)

        last_hidden = gru_out[:, -1, :]
        out = self.fc(last_hidden)

        return out

def main():
    """
    Simple sanity check: create a model and pass a dummy batch through it.

    Run with:
        python model.py
    """
    batch_size = 32
    seq_len = 48
    input_size = 1
    horizon = 1

    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        horizon=horizon,
    )

    print(model)

    # Fake input batch: normally this would come from your DataLoader
    X_dummy = torch.randn(batch_size, seq_len, input_size)  # (32, 48, 1)

    y_pred = model(X_dummy)

    print("Input shape: ", X_dummy.shape)
    print("Output shape:", y_pred.shape)  # should be (32, horizon)


if __name__ == "__main__":
    main()
