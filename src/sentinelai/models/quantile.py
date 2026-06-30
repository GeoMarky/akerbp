"""1D-CNN quantile regressor — predicts multiple quantiles of a target sensor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sentinelai.data.windows import WindowBatch


class CNNQuantileRegressor(nn.Module):
    """Predict K quantiles of the target value at the end of each window."""

    def __init__(self, n_channels: int, n_quantiles: int, window_length: int = 60):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.backbone = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss across quantiles."""
    # preds: (batch, K), target: (batch,), quantiles: (K,)
    errors = target.unsqueeze(1) - preds
    loss = torch.maximum(quantiles * errors, (quantiles - 1) * errors)
    return loss.mean()


@dataclass
class QuantileTrainResult:
    model: CNNQuantileRegressor
    quantile_levels: np.ndarray
    train_losses: list[float]


def build_regression_windows(
    batch: WindowBatch,
    target_sensor_idx: int,
    input_sensor_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X: value+mask+dt for input sensors only.
    y: target sensor value at last timestep of window.
    coverage: per-window coverage on input sensors.
    """
    vals = batch.values[:, input_sensor_indices, :]
    masks = batch.mask[:, input_sensor_indices, :]
    dts = batch.dt[:, input_sensor_indices, :]
    x = np.concatenate([vals, masks, dts], axis=1).astype(np.float32)
    y = batch.values[:, target_sensor_idx, -1].astype(np.float32)
    cov = batch.mask[:, input_sensor_indices, :].mean(axis=(1, 2))
    return x, y, cov


def train_quantile_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
) -> QuantileTrainResult:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    q_arr = np.array(quantile_levels, dtype=np.float32)
    q_tensor = torch.tensor(q_arr, device=device_t)

    model = CNNQuantileRegressor(x_train.shape[1], len(quantile_levels)).to(device_t)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device_t)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device_t)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch = 0.0
        for xb, yb in loader:
            optim.zero_grad()
            pred = model(xb)
            loss = pinball_loss(pred, yb, q_tensor)
            loss.backward()
            optim.step()
            epoch += loss.item() * len(xb)
        losses.append(epoch / len(xt))

    return QuantileTrainResult(model=model, quantile_levels=q_arr, train_losses=losses)


@torch.no_grad()
def predict_quantiles(
    model: CNNQuantileRegressor,
    x: np.ndarray,
    quantile_levels: np.ndarray,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device_t)
    preds = model(xt).cpu().numpy()
    out = {f"q{int(q*100)}": preds[:, i] for i, q in enumerate(quantile_levels)}
    out["lower"] = preds[:, 0]
    out["median"] = preds[:, len(quantile_levels) // 2]
    out["upper"] = preds[:, -1]
    return out
