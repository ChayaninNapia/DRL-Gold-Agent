from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path("data") / "GOLD_M1_last365_trading_days_to_2026-05-01.parquet"
OUTPUT_FILE = Path("data") / "GOLD_M1_short_days_by_weekday.png"
FULL_DAY_COUNTS = {1378, 1379}
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def main():
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    if "time" not in df.columns:
        raise SystemExit("Input file must contain a 'time' column.")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    daily = (
        df.groupby(df["time"].dt.date)
        .size()
        .rename("candles")
        .reset_index()
    )
    daily = daily.rename(columns={"time": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["weekday"] = daily["date"].dt.day_name()

    short_days = daily[~daily["candles"].isin(FULL_DAY_COUNTS)].copy()
    weekday_counts = (
        short_days["weekday"]
        .value_counts()
        .reindex(WEEKDAY_ORDER, fill_value=0)
    )

    print(f"Loaded {len(df)} candles from {INPUT_FILE}")
    print(f"Total trading days: {len(daily)}")
    print(f"Short days excluding {sorted(FULL_DAY_COUNTS)}: {len(short_days)}")
    print()
    print("Short days by weekday:")
    print(weekday_counts.to_string())
    print()
    print("Short-day details:")
    print(
        short_days[["date", "weekday", "candles"]]
        .sort_values(["date"])
        .to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4c78a8" if count == 0 else "#f58518" for count in weekday_counts]
    bars = ax.bar(weekday_counts.index, weekday_counts.values, color=colors, edgecolor="black")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.15,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Short GOLD M1 Trading Days by Weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Number of short days")
    ax.set_ylim(0, max(weekday_counts.max() + 2, 3))
    ax.grid(axis="y", alpha=0.25)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150)
    plt.close(fig)

    print()
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    raise SystemExit(main())
