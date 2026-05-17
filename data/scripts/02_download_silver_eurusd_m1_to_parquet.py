"""Download SILVER and EURUSD M1 data (750 trading days) from MT5 to parquet.

Run from repo root:
    & 'd:\EA\.venv\Scripts\python.exe' 'data/scripts/02_download_silver_eurusd_m1_to_parquet.py'
"""
from datetime import datetime, timedelta, timezone
from importlib.util import find_spec
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


DATE_TO = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
TARGET_TRADING_DAYS = 750
LOOKBACK_CALENDAR_DAYS = 1800  # ~5 years calendar buffer

# (mt5_symbol, output_prefix)  — filename gets _to_<last_date> appended at runtime
SYMBOLS = [
    ("SILVER", "SILVER_M1_last750_trading_days"),
    ("EURUSD", "EURUSD_M1_last750_trading_days"),
]

OUTPUT_DIR = Path("data")


def download_symbol(symbol: str, output_prefix: str) -> int:
    print(f"\n{'='*60}", flush=True)
    print(f"Checking symbol {symbol}...", flush=True)
    symbol_info = mt5.symbol_info(symbol)

    if symbol_info is None:
        print(f"  Symbol not found: {symbol}", flush=True)
        print(f"  Try alternative names (e.g. XAGUSD for Silver, XAGUSDm, etc.)", flush=True)
        return 1

    if not symbol_info.visible:
        print(f"  Selecting {symbol} in Market Watch...", flush=True)
        if not mt5.symbol_select(symbol, True):
            print(f"  Failed to select symbol: {symbol}", flush=True)
            return 1

    download_from = DATE_TO - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    print(
        f"  Downloading {symbol} M1  {download_from:%Y-%m-%d} to {DATE_TO:%Y-%m-%d}...",
        flush=True,
    )
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, download_from, DATE_TO)

    if rates is None or len(rates) == 0:
        print(f"  No data received: {mt5.last_error()}", flush=True)
        return 1

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["trading_date"] = df["time"].dt.date

    trading_dates = pd.Index(df["trading_date"].unique()).sort_values()
    if len(trading_dates) < TARGET_TRADING_DAYS:
        print(
            f"  Only {len(trading_dates)} trading days found "
            f"(need {TARGET_TRADING_DAYS}). Increase LOOKBACK_CALENDAR_DAYS.",
            flush=True,
        )
        return 1

    selected_dates = trading_dates[-TARGET_TRADING_DAYS:]
    df = df[df["trading_date"].isin(selected_dates)].copy()
    df = df.drop(columns=["trading_date"]).reset_index(drop=True)

    last_date = str(selected_dates[-1])
    output_file = OUTPUT_DIR / f"{output_prefix}_to_{last_date}.parquet"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)

    print(f"  Rows:          {len(df):,}", flush=True)
    print(f"  Trading days:  {len(selected_dates)}", flush=True)
    print(f"  First date:    {selected_dates[0]}", flush=True)
    print(f"  Last date:     {selected_dates[-1]}", flush=True)
    print(f"  First candle:  {df['time'].iloc[0]}", flush=True)
    print(f"  Last candle:   {df['time'].iloc[-1]}", flush=True)
    print(f"  Saved to:      {output_file}", flush=True)
    return 0


def main() -> int:
    if find_spec("pyarrow") is None and find_spec("fastparquet") is None:
        raise SystemExit(
            "Missing parquet engine. Install:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install pyarrow"
        )

    print("Connecting to MT5...", flush=True)
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error(), flush=True)
        return 1

    print(f"MT5 version: {mt5.version()}", flush=True)

    errors = 0
    for symbol, stem in SYMBOLS:
        errors += download_symbol(symbol, stem)

    mt5.shutdown()

    print(f"\n{'='*60}", flush=True)
    if errors == 0:
        print("All symbols downloaded successfully.", flush=True)
    else:
        print(f"{errors}/{len(SYMBOLS)} symbol(s) failed. Check output above.", flush=True)

    return errors


if __name__ == "__main__":
    raise SystemExit(main())
