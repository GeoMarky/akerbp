"""1D-CNN heteroscedastic regressor — predicts mean and log-variance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sentinelai.models.quantile import build_regression_windows


class CNNHeteroscedasticRegressor(nn.Module):
    """Gaussian head: mean and log-variance for the target at window end."""

    def __init__(self, n_channels: int, window_length: int = 60):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mean_head(h).squeeze(-1), self.logvar_head(h).squeeze(-1)


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood."""
    var = torch.exp(logvar).clamp(min=1e-6)
    return (0.5 * (logvar + (target - mean) ** 2 / var)).mean()


@dataclass
class HeteroTrainResult:
    model: CNNHeteroscedasticRegressor
    train_losses: list[float]


def train_heteroscedastic_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
) -> HeteroTrainResult:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CNNHeteroscedasticRegressor(x_train.shape[1]).to(device_t)
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
            mean, logvar = model(xb)
            loss = gaussian_nll(mean, logvar, yb)
            loss.backward()
            optim.step()
            epoch += loss.item() * len(xb)
        losses.append(epoch / len(xt))

    return HeteroTrainResult(model=model, train_losses=losses)


@torch.no_grad()
def predict_interval(
    model: CNNHeteroscedasticRegressor,
    x: np.ndarray,
    z: float = 1.645,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Return mean and approximate 90% interval (z=1.645)."""
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device_t)
    mean, logvar = model(xt)
    mean_np = mean.cpu().numpy()
    std_np = np.exp(0.5 * logvar.cpu().numpy())
    return {
        "mean": mean_np,
        "std": std_np,
        "lower": mean_np - z * std_np,
        "upper": mean_np + z * std_np,
    }
