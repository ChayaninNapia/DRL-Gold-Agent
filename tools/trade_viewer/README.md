# DRL Trade Viewer

A local Vite + React + TypeScript app with two tabs (dark theme):

- **View Parquet** — inspect MetaTrader 5 GOLD M1 parquet exports (table, line/scatter plots, schema).
- **Trade History** — candlestick chart (M1/M5/M15/M30/H1) of an OHLC parquet with entry/exit
  markers overlaid from a run's `trades.csv`. Long entry = blue up-triangle, short entry = red
  down-triangle, exits = matching-color X. Scroll to zoom, drag to pan, double-click or the
  Reset View button to reset. Filter by phase and episode.

## Install

```bash
npm install
```

## Run

From `d:\EA\tools\trade_viewer`:

```powershell
npm install   # first time only
npm run dev
```

Then open **http://127.0.0.1:5173** in your browser (Vite is pinned to host
`127.0.0.1`, port `5173` in `vite.config.ts`).

- **View Parquet** tab: upload a `.parquet` file.
- **Trade History** tab: upload an OHLC parquet (e.g.
  `d:\EA\data\GOLD_M1_last750_trading_days_to_2026-05-01.parquet`) **and** a
  `trades.csv` from a run (e.g. `d:\EA\runs\final_<algo>_s<seed>\trades.csv`),
  then pick timeframe / phase / episode.

## Test and Build

```bash
npm test
npm run build
```

## Expected Columns

The app is tuned for files exported from `MetaTrader5.copy_rates_range()` through pandas:

- `time`
- `open`
- `high`
- `low`
- `close`
- `tick_volume`
- `spread`
- `real_volume`

It will still open other parquet files, but it shows a warning when expected MT5 columns are missing.

## Performance Notes

- Parquet reading is browser-side with `hyparquet` and `hyparquet-compressors`.
- Charts are downsampled before rendering. The default chart cap is 5,000 points.
- Min/max calculations use loops, not `Math.min(...largeArray)` or `Math.max(...largeArray)`, to avoid call stack overflows on large M1 datasets.
- Raw numeric values stay numeric for plotting. Decimal formatting is only applied in display layers such as tables and tooltips.

## Why DuckDB WASM Is Not Used

DuckDB WASM can require a runtime download of parquet extensions from `extensions.duckdb.org`. This app avoids that path so it works locally without the known parquet extension fetch failure.
