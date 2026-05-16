from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path("data") / "GOLD_M1_last365_trading_days_to_2026-05-01.parquet"
HEATMAP_FILE = Path("data") / "GOLD_M1_missing_minutes_lt1377_heatmap.png"
HOURLY_FILE = Path("data") / "GOLD_M1_missing_minutes_lt1377_by_hour.png"
SHORT_DAY_THRESHOLD = 1377
REFERENCE_FULL_DAY_CANDLES = 1379


def main():
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    if "time" not in df.columns:
        raise SystemExit("Input file must contain a 'time' column.")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.date
    df["minute_of_day"] = df["time"].dt.hour * 60 + df["time"].dt.minute

    daily_counts = df.groupby("date").size()
    short_dates = daily_counts[daily_counts < SHORT_DAY_THRESHOLD].index

    reference_dates = daily_counts[daily_counts == REFERENCE_FULL_DAY_CANDLES].index
    if len(reference_dates) == 0:
        raise SystemExit(f"No {REFERENCE_FULL_DAY_CANDLES}-candle days found for reference.")

    reference_minutes = np.array(
        sorted(df[df["date"].isin(reference_dates)]["minute_of_day"].unique())
    )

    if len(short_dates) == 0:
        raise SystemExit(f"No days found with candles/day < {SHORT_DAY_THRESHOLD}.")

    missing_rows = []
    row_labels = []
    for date in short_dates:
        present_minutes = set(df.loc[df["date"] == date, "minute_of_day"])
        missing_rows.append([minute not in present_minutes for minute in reference_minutes])
        row_labels.append(f"{date} ({daily_counts.loc[date]})")

    missing = np.array(missing_rows, dtype=int)
    missing_by_hour = pd.Series(0, index=range(24), dtype=int)
    for minute, total_missing in zip(reference_minutes, missing.sum(axis=0)):
        missing_by_hour.loc[minute // 60] += int(total_missing)

    print(f"Loaded {len(df)} candles from {INPUT_FILE}")
    print(f"Reference full days: {len(reference_dates)} days with {REFERENCE_FULL_DAY_CANDLES} candles")
    print(f"Reference minutes/day: {len(reference_minutes)}")
    print(f"Short days with candles/day < {SHORT_DAY_THRESHOLD}: {len(short_dates)}")
    print()
    print("Short days:")
    print(daily_counts.loc[short_dates].to_string())
    print()
    print("Missing reference minutes by UTC hour:")
    print(missing_by_hour.to_string())

    fig_height = max(6, len(row_labels) * 0.35)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.imshow(missing, aspect="auto", cmap="Reds", interpolation="nearest")

    hour_ticks = []
    hour_labels = []
    for hour in range(24):
        minute = hour * 60
        idx = np.searchsorted(reference_minutes, minute)
        if idx < len(reference_minutes):
            hour_ticks.append(idx)
            hour_labels.append(f"{hour:02d}:00")

    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(f"Missing GOLD M1 Minutes on Days With < {SHORT_DAY_THRESHOLD} Candles")
    ax.set_xlabel("UTC time of day")
    ax.set_ylabel("Date (candles/day)")

    fig.tight_layout()
    HEATMAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HEATMAP_FILE, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [f"{hour:02d}" for hour in missing_by_hour.index],
        missing_by_hour.values,
        color="#f58518",
        edgecolor="black",
    )
    ax.set_title(f"Missing GOLD M1 Reference Minutes by UTC Hour (< {SHORT_DAY_THRESHOLD} Candle Days)")
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("Missing minutes across short days")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(HOURLY_FILE, dpi=150)
    plt.close(fig)

    print()
    print(f"Saved heatmap to {HEATMAP_FILE}")
    print(f"Saved hourly summary to {HOURLY_FILE}")


if __name__ == "__main__":
    raise SystemExit(main())
