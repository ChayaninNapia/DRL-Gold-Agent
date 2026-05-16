from datetime import datetime, timedelta, timezone
from importlib.util import find_spec
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


SYMBOL = "GOLD"
TIMEFRAME = mt5.TIMEFRAME_M1
DATE_TO = datetime(2026, 5, 1, tzinfo=timezone.utc)
TARGET_TRADING_DAYS = 750
LOOKBACK_CALENDAR_DAYS = 1200
OUTPUT_FILE = Path("data") / "GOLD_M1_last750_trading_days_to_2026-05-01.parquet"


def main():
    # Fail before connecting to MT5 if pandas cannot write parquet files.
    if find_spec("pyarrow") is None and find_spec("fastparquet") is None:
        raise SystemExit(
            "Missing parquet engine. Install one with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install pyarrow"
        )

    print("Connecting to MT5...", flush=True)
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error(), flush=True)
        return 1

    print(f"Checking symbol {SYMBOL}...", flush=True)
    symbol_info = mt5.symbol_info(SYMBOL)

    if symbol_info is None:
        print(f"Symbol not found: {SYMBOL}", flush=True)
        mt5.shutdown()
        return 1

    if not symbol_info.visible:
        print(f"Selecting symbol {SYMBOL}...", flush=True)
        if not mt5.symbol_select(SYMBOL, True):
            print(f"Failed to select symbol: {SYMBOL}", flush=True)
            mt5.shutdown()
            return 1

    download_from = DATE_TO - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    print(
        f"Downloading {SYMBOL} M1 candles from "
        f"{download_from:%Y-%m-%d %H:%M:%S %Z} to {DATE_TO:%Y-%m-%d %H:%M:%S %Z}...",
        flush=True,
    )
    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, download_from, DATE_TO)

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("No data received from MT5:", mt5.last_error(), flush=True)
        return 1

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["trading_date"] = df["time"].dt.date

    trading_dates = pd.Index(df["trading_date"].unique()).sort_values()
    if len(trading_dates) < TARGET_TRADING_DAYS:
        print(
            f"Only found {len(trading_dates)} trading days, "
            f"but need {TARGET_TRADING_DAYS}. Increase LOOKBACK_CALENDAR_DAYS.",
            flush=True,
        )
        return 1

    selected_dates = trading_dates[-TARGET_TRADING_DAYS:]
    df = df[df["trading_date"].isin(selected_dates)].copy()
    df = df.drop(columns=["trading_date"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_FILE}", flush=True)
    print(f"Trading days: {len(selected_dates)}", flush=True)
    print(f"First trading date: {selected_dates[0]}", flush=True)
    print(f"Last trading date:  {selected_dates[-1]}", flush=True)
    print(f"First candle: {df['time'].iloc[0]}", flush=True)
    print(f"Last candle:  {df['time'].iloc[-1]}", flush=True)
    print(df.tail())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
