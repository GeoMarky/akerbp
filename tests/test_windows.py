"""Tests for windowing and gap handling."""

import numpy as np
import pandas as pd

from sentinelai.config import SENSORS
from sentinelai.data.windows import inject_gaps, make_windows, stack_channels, temporal_split


def _toy_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {s: rng.normal(size=n) for s in SENSORS}
    data["datetime"] = pd.date_range("2020-01-01", periods=n, freq="s")
    data["anomaly"] = np.zeros(n, dtype=int)
    data["anomaly"][150:] = 1
    return pd.DataFrame(data)


def test_make_windows_shape():
    df = _toy_df()
    batch = make_windows(df, window_length=20, window_step=10)
    n_sensors = len(SENSORS)
    assert batch.values.shape[1] == n_sensors
    assert batch.values.shape[2] == 20
    assert batch.mask.shape == batch.values.shape
    assert batch.dt.shape == batch.values.shape
    assert len(batch.labels) == len(batch.coverage)


def test_stack_channels_triples_sensors():
    df = _toy_df()
    batch = make_windows(df, window_length=20, window_step=10)
    stacked = stack_channels(batch)
    assert stacked.shape[1] == len(SENSORS) * 3


def test_inject_gaps_reduces_coverage():
    df = _toy_df()
    gapped = inject_gaps(df, gap_prob=0.2, correlated_prob=1.0)
    batch = make_windows(gapped, window_length=20, window_step=10)
    assert batch.coverage.min() < 1.0
    assert batch.mask.min() == 0.0


def test_temporal_split_order():
    df = _toy_df(300)
    batch = make_windows(df, window_length=20, window_step=10)
    train, cal, test = temporal_split(batch)
    assert len(train.labels) + len(cal.labels) + len(test.labels) == len(batch.labels)
    assert train.indices[-1] < cal.indices[0]
    assert cal.indices[-1] < test.indices[0]
