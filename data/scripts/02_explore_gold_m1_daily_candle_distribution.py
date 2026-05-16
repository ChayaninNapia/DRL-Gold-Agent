from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path("data") / "GOLD_M1_2025-05-01_to_2026-05-01.parquet"
OUTPUT_FILE = Path("data") / "GOLD_M1_daily_candle_count_distribution.png"


def main():
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    if "time" not in df.columns:
        raise SystemExit("Input file must contain a 'time' column.")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.date

    daily_counts = df.groupby("date").size().rename("candles")

    print(f"Loaded {len(df)} candles from {INPUT_FILE}")
    print(f"Daily count rows: {len(daily_counts)} days")
    print()
    print(daily_counts.describe())
    print()
    print("Lowest candle-count days:")
    print(daily_counts.sort_values().head(10))
    print()
    print("Highest candle-count days:")
    print(daily_counts.sort_values(ascending=False).head(10))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        daily_counts,
        bins=range(int(daily_counts.min()), int(daily_counts.max()) + 2),
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axvline(
        daily_counts.mean(),
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {daily_counts.mean():.1f}",
    )
    ax.set_title("Distribution of M1 Candles per Day")
    ax.set_xlabel("Number of candles in one day")
    ax.set_ylabel("Number of days")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150)
    plt.close(fig)

    print()
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    raise SystemExit(main())
