"""TimesFM 2.0 hazard detector: frozen backbone + trainable classification head."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from sentinelai.config import SENSORS, WINDOW_LENGTH
from sentinelai.data.windows import WindowBatch
from sentinelai.models.timesfm_backbone import PatchedTimeSeriesDecoder, TimesFMConfig

TIMESFM_REPO_ID = "google/timesfm-2.0-500m-pytorch"
TIMESFM_CKPT = "torch_model.ckpt"
PATCH_LEN = 32
HIDDEN_SIZE = 1280
N_SENSORS = len(SENSORS)
EMBED_DIM = N_SENSORS * HIDDEN_SIZE


def _padded_length(window_length: int = WINDOW_LENGTH) -> int:
    if window_length % PATCH_LEN == 0:
        return window_length
    return ((window_length // PATCH_LEN) + 1) * PATCH_LEN


def pool_sensor_embeddings(
    embeddings: np.ndarray,
    mode: str,
    n_sensors: int = N_SENSORS,
) -> np.ndarray:
    """
    Collapse per-sensor embeddings by pooling instead of concatenating.

    ``embed_windows`` concatenates the per-sensor embeddings into a
    ``(n_windows, n_sensors * HIDDEN_SIZE)`` array (sensor-major order). This
    reshapes back to ``(n_windows, n_sensors, HIDDEN_SIZE)`` and pools across
    sensors, yielding ``(n_windows, HIDDEN_SIZE)`` — an ``n_sensors``x width
    reduction that regularizes the head on wide inputs.
    """
    n_windows, embed_dim = embeddings.shape
    if embed_dim % n_sensors != 0:
        raise ValueError(
            f"Embedding width {embed_dim} is not divisible by n_sensors {n_sensors}; "
            "pooling expects concatenated per-sensor embeddings."
        )
    per_sensor = embeddings.reshape(n_windows, n_sensors, embed_dim // n_sensors)
    if mode == "mean":
        return per_sensor.mean(axis=1)
    if mode == "max":
        return per_sensor.max(axis=1)
    raise ValueError(f"Unknown sensor_pool mode: {mode!r} (expected 'mean' or 'max')")


def load_timesfm_backbone(device: str | None = None) -> PatchedTimeSeriesDecoder:
    """Load pretrained TimesFM 2.0 (500M) and freeze all backbone weights."""
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = TimesFMConfig(
        num_layers=50,
        num_heads=16,
        num_kv_heads=16,
        hidden_size=1280,
        intermediate_size=1280,
        head_dim=80,
        patch_len=PATCH_LEN,
        horizon_len=128,
        use_positional_embedding=False,
    )
    backbone = PatchedTimeSeriesDecoder(config)
    ckpt_path = hf_hub_download(TIMESFM_REPO_ID, TIMESFM_CKPT)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone.to(device_t)


def _build_padding(
    batch: WindowBatch,
    sensor_idx: int,
    use_mask: bool,
    padded_len: int,
) -> np.ndarray:
    """Build TimesFM input_padding (1 = ignore token) for one sensor channel."""
    n_windows = len(batch.labels)
    window_len = batch.values.shape[2]
    pad = np.zeros((n_windows, padded_len), dtype=np.float32)
    pad[:, window_len:] = 1.0
    if use_mask:
        pad[:, :window_len] = 1.0 - batch.mask[:, sensor_idx, :]
    return pad


@torch.no_grad()
def embed_windows(
    backbone: PatchedTimeSeriesDecoder,
    batch: WindowBatch,
    use_mask: bool = False,
    batch_size: int = 512,
    device: str | None = None,
) -> np.ndarray:
    """
    Extract multivariate window embeddings from the frozen TimesFM backbone.

    Each sensor is encoded independently; patch embeddings are mean-pooled over
    valid (non-padded) patches and concatenated across sensors.
    """
    device_t = torch.device(device or next(backbone.parameters()).device)
    backbone.eval()
    padded_len = _padded_length(batch.values.shape[2])
    n_windows, n_sensors, _ = batch.values.shape
    all_embeddings: list[np.ndarray] = []

    for sensor_idx in range(n_sensors):
        values = batch.values[:, sensor_idx, :]
        if padded_len > values.shape[1]:
            pad_values = np.zeros((n_windows, padded_len - values.shape[1]), dtype=np.float32)
            values = np.concatenate([values, pad_values], axis=1)

        padding = _build_padding(batch, sensor_idx, use_mask, padded_len)
        sensor_embeddings: list[np.ndarray] = []

        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            x = torch.tensor(values[start:end], dtype=torch.float32, device=device_t)
            p = torch.tensor(padding[start:end], dtype=torch.float32, device=device_t)
            freq = torch.zeros(end - start, dtype=torch.long, device=device_t)

            patch_emb, patch_pad = backbone.encode(x, p, freq)
            valid = (1.0 - patch_pad).unsqueeze(-1)
            pooled = (patch_emb * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            sensor_embeddings.append(pooled.cpu().numpy())

        all_embeddings.append(np.concatenate(sensor_embeddings, axis=0))

    return np.concatenate(all_embeddings, axis=1)


class TimesFMClassifier(nn.Module):
    """MLP classification head on concatenated TimesFM sensor embeddings."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_size: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainResult:
    model: TimesFMClassifier
    train_losses: list[float]
    val_losses: list[float]
    scaler: StandardScaler | None = None
    sensor_pool: str | None = None


def _to_tensor(
    embeddings: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(embeddings, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return x, y


def train_detector(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_size: int = 128,
    dropout: float = 0.2,
    normalize: bool = False,
    sensor_pool: str | None = None,
    device: str | None = None,
) -> TrainResult:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Pool per-sensor embeddings (mean/max) before anything else so the scaler
    # and head see the reduced width.
    if sensor_pool is not None:
        train_emb = pool_sensor_embeddings(train_emb, sensor_pool)
        if val_emb is not None:
            val_emb = pool_sensor_embeddings(val_emb, sensor_pool)

    model = TimesFMClassifier(train_emb.shape[1], hidden_size, dropout).to(device_t)

    # Standardize the wide TimesFM embeddings; fit on train only to avoid leakage.
    scaler: StandardScaler | None = None
    if normalize:
        scaler = StandardScaler().fit(train_emb)
        train_emb = scaler.transform(train_emb)
        if val_emb is not None:
            val_emb = scaler.transform(val_emb)

    x_train, y_train = _to_tensor(train_emb, train_labels, device_t)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    pos = max(y_train.sum().item(), 1)
    neg = max(len(y_train) - pos, 1)
    weight = torch.tensor([1.0, neg / pos], device=device_t)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", patience=5)

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

        if val_emb is not None and val_labels is not None:
            model.eval()
            with torch.no_grad():
                xv, yv = _to_tensor(val_emb, val_labels, device_t)
                vl = criterion(model(xv), yv).item()
            val_losses.append(vl)
            scheduler.step(vl)
        else:
            scheduler.step(train_loss)

    return TrainResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        scaler=scaler,
        sensor_pool=sensor_pool,
    )


@torch.no_grad()
def predict_logits(
    model: TimesFMClassifier,
    embeddings: np.ndarray,
    scaler: StandardScaler | None = None,
    sensor_pool: str | None = None,
    device: str | None = None,
) -> np.ndarray:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    if sensor_pool is not None:
        embeddings = pool_sensor_embeddings(embeddings, sensor_pool)
    if scaler is not None:
        embeddings = scaler.transform(embeddings)
    x = torch.tensor(embeddings, dtype=torch.float32, device=device_t)
    return model(x).cpu().numpy()


@torch.no_grad()
def predict_proba(
    model: TimesFMClassifier,
    embeddings: np.ndarray,
    scaler: StandardScaler | None = None,
    sensor_pool: str | None = None,
    device: str | None = None,
) -> np.ndarray:
    logits = predict_logits(
        model, embeddings, scaler=scaler, sensor_pool=sensor_pool, device=device
    )
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs[:, 1]
