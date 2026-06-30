"""1D-CNN hazard detector with value + mask + dt input channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sentinelai.data.windows import WindowBatch, stack_channels


class CNNDetector(nn.Module):
    """
    1D CNN over multivariate windows.

    The convolutional encoder mirrors the SKAB ``Conv_AE`` design
    (https://github.com/waico/SKAB/blob/master/core/Conv_AE.py): two strided
    ``kernel_size=7`` convolutions with a ``32 -> 16`` filter progression and
    ``Dropout(0.2)`` between them. Where ``Conv_AE`` decodes back to a
    reconstruction, we instead pool the encoded features into a classification
    head, since the rest of the pipeline consumes hazard logits.

    Input shape: (batch, 3 * n_sensors, window_length)
    Output: logits (batch, 2) for [nominal, hazard]
    """

    def __init__(self, n_channels: int, window_length: int = 60):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainResult:
    model: CNNDetector
    train_losses: list[float]
    val_losses: list[float]


def _stack_input(batch: WindowBatch, values_only: bool = False) -> np.ndarray:
    """Values-only (weak baseline) or value + mask + dt (improved path)."""
    if values_only:
        return batch.values
    return stack_channels(batch)


def _to_tensor(
    batch: WindowBatch,
    device: torch.device,
    values_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(_stack_input(batch, values_only), dtype=torch.float32, device=device)
    y = torch.tensor(batch.labels, dtype=torch.long, device=device)
    return x, y


def train_detector(
    train: WindowBatch,
    val: WindowBatch | None = None,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    values_only: bool = False,
) -> TrainResult:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_sensors = train.values.shape[1]
    n_channels = n_sensors if values_only else n_sensors * 3
    window_length = train.values.shape[2]
    model = CNNDetector(n_channels, window_length).to(device_t)

    x_train, y_train = _to_tensor(train, device_t, values_only=values_only)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    # Class weights for imbalanced labels
    pos = max(y_train.sum().item(), 1)
    neg = max(len(y_train) - pos, 1)
    weight = torch.tensor([1.0, neg / pos], device=device_t)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", patience=5
    )

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(xb)
        train_loss = epoch_loss / len(x_train)
        train_losses.append(train_loss)

        if val is not None:
            model.eval()
            with torch.no_grad():
                xv, yv = _to_tensor(val, device_t, values_only=values_only)
                vl = criterion(model(xv), yv).item()
            val_losses.append(vl)
            scheduler.step(vl)
        else:
            scheduler.step(train_loss)

    return TrainResult(model=model, train_losses=train_losses, val_losses=val_losses)


@torch.no_grad()
def predict_logits(
    model: CNNDetector,
    batch: WindowBatch,
    device: str | None = None,
    values_only: bool = False,
) -> np.ndarray:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    x, _ = _to_tensor(batch, device_t, values_only=values_only)
    logits = model(x).cpu().numpy()
    return logits


@torch.no_grad()
def predict_proba(
    model: CNNDetector,
    batch: WindowBatch,
    device: str | None = None,
    values_only: bool = False,
) -> np.ndarray:
    logits = predict_logits(model, batch, device=device, values_only=values_only)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    # #region agent log
    import json, time
    with open("/workspaces/solution/.cursor/debug-4639f7.log", "a") as _f:
        _f.write(json.dumps({"sessionId": "4639f7", "hypothesisId": "B", "location": "cnn_detector.py:predict_proba", "message": "predict_proba shapes", "data": {"logits_shape": list(logits.shape), "probs_out_shape": list(probs[:, 1].shape)}, "timestamp": int(time.time() * 1000), "runId": "pre-fix"}) + "\n")
    # #endregion
    return probs[:, 1]
