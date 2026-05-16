"""Market features per proposal §4.2.

10 market dims:  5 close-return windows + MACD + STO + ATR + RSI + spread
All causal (no future leakage). NaNs in the warmup window are filled with 0
after computation so the env can use the very first bar safely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_COLUMNS = [
    "ret_1", "ret_5", "ret_15", "ret_30", "ret_60",
    "macd", "sto", "rsi", "atr", "spread_pts",
]


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(n, min_periods=n).mean()
    avg_loss = loss.rolling(n, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    hn = high.rolling(n, min_periods=n).max()
    ln = low.rolling(n, min_periods=n).min()
    sto = 100.0 * (close - ln) / (hn - ln).replace(0.0, np.nan)
    return sto


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute features per row. df is assumed to have columns:
    time, open, high, low, close, spread, session_date, ...

    Returns a copy of df with new feature columns. Features are computed
    over the WHOLE series so warmup uses real history across day boundaries;
    that is fine because all rolling/EWMA windows are causal.
    """
    out = df.copy()
    c = out["close"]

    for w in cfg["return_windows"]:
        out[f"ret_{w}"] = c.pct_change(w)

    ema_s = c.ewm(span=cfg["macd_short"], adjust=False).mean()
    ema_l = c.ewm(span=cfg["macd_long"], adjust=False).mean()
    out["macd"] = ema_s - ema_l

    out["sto"] = _stoch(out["high"], out["low"], out["close"], cfg["sto_window"])
    out["rsi"] = _rsi(out["close"], cfg["rsi_window"])
    out["atr"] = _atr(out["high"], out["low"], out["close"], cfg["atr_window"])

    # spread in raw points (broker units, e.g. 1 point = 0.01 USD on XAUUSD MT5).
    # Keep as a feature; cost in env uses commission, not this column.
    out["spread_pts"] = out["spread"].astype("float64")

    for col in MARKET_COLUMNS:
        out[col] = out[col].astype("float64").fillna(0.0)

    return out


if __name__ == "__main__":
    from pathlib import Path
    import yaml

    from data import load_raw, select_window

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8"))
    df = load_raw(cfg["data"]["path"])
    df = select_window(df, cfg["data"]["window_days"])
    feat = build_features(df, cfg["features"])
    print(feat[MARKET_COLUMNS].describe().T)
