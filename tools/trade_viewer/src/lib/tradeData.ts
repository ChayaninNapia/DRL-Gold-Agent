import { readParquetFile } from './parquet';
import { toNumeric, toTimestamp } from './formatting';
import type { LoadedParquet, ParquetRow } from '../types/parquet';

// ---------------------------------------------------------------------------
// Timeframes
// ---------------------------------------------------------------------------

export type Timeframe = 'M1' | 'M5' | 'M15' | 'M30' | 'H1';

export const TIMEFRAMES: Timeframe[] = ['M1', 'M5', 'M15', 'M30', 'H1'];

const TF_MINUTES: Record<Timeframe, number> = {
  M1: 1,
  M5: 5,
  M15: 15,
  M30: 30,
  H1: 60,
};

// ---------------------------------------------------------------------------
// OHLC bars
// ---------------------------------------------------------------------------

export interface Bar {
  t: number; // epoch ms (bucket start)
  o: number;
  h: number;
  l: number;
  c: number;
}

/** Extract M1 OHLC bars from a loaded parquet (expects time/open/high/low/close). */
export function barsFromParquet(loaded: LoadedParquet): Bar[] {
  const cols = loaded.columns.map((c) => c.toLowerCase());
  const idx = (name: string) => cols.indexOf(name);
  const find = (row: ParquetRow, name: string) => {
    const original = loaded.columns[idx(name)];
    return original !== undefined ? row[original] : undefined;
  };
  if (idx('time') < 0 || idx('open') < 0 || idx('high') < 0 || idx('low') < 0 || idx('close') < 0) {
    throw new Error('Parquet must contain time, open, high, low, close columns.');
  }

  const bars: Bar[] = [];
  for (const row of loaded.rows) {
    const t = toTimestamp(find(row, 'time'));
    const o = toNumeric(find(row, 'open'));
    const h = toNumeric(find(row, 'high'));
    const l = toNumeric(find(row, 'low'));
    const c = toNumeric(find(row, 'close'));
    if (t === null || o === null || h === null || l === null || c === null) continue;
    bars.push({ t, o, h, l, c });
  }
  bars.sort((a, b) => a.t - b.t);
  return bars;
}

/** Resample M1 bars into a coarser timeframe by floor-bucketing on epoch ms. */
export function resampleBars(m1: Bar[], tf: Timeframe): Bar[] {
  if (tf === 'M1') return m1;
  const bucketMs = TF_MINUTES[tf] * 60_000;
  const out: Bar[] = [];
  let cur: Bar | null = null;
  let curBucket = -1;
  for (const bar of m1) {
    const bucket = Math.floor(bar.t / bucketMs) * bucketMs;
    if (cur === null || bucket !== curBucket) {
      if (cur) out.push(cur);
      cur = { t: bucket, o: bar.o, h: bar.h, l: bar.l, c: bar.c };
      curBucket = bucket;
    } else {
      cur.h = Math.max(cur.h, bar.h);
      cur.l = Math.min(cur.l, bar.l);
      cur.c = bar.c;
    }
  }
  if (cur) out.push(cur);
  return out;
}

export async function loadOhlcParquet(file: File): Promise<{ loaded: LoadedParquet; m1: Bar[] }> {
  const loaded = await readParquetFile(file);
  const m1 = barsFromParquet(loaded);
  if (m1.length === 0) throw new Error('No usable OHLC rows found in parquet.');
  return { loaded, m1 };
}

// ---------------------------------------------------------------------------
// Trades CSV
// ---------------------------------------------------------------------------

export interface Trade {
  group: string; // episode (DRL) or baseline name
  phase: string; // train | val | test
  day: string;
  entryTime: number; // epoch ms
  exitTime: number;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  barsHeld: number;
  pnlLog: number;
}

function splitCsvLine(line: string): string[] {
  // trades.csv from the trainers has no quoted fields, but keep a minimal
  // quote-aware splitter in case timestamps are ever quoted.
  const out: string[] = [];
  let cur = '';
  let inQ = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      inQ = !inQ;
    } else if (ch === ',' && !inQ) {
      out.push(cur);
      cur = '';
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

/**
 * Parse a trades.csv timestamp as UTC.
 *
 * The trainers write entry_time/exit_time via pandas
 * `dt.strftime("%Y-%m-%d %H:%M:%S")` on a UTC datetime column, producing e.g.
 * "2026-01-15 01:03:00" with NO timezone suffix. JavaScript's Date.parse()
 * treats a space-separated, suffix-less datetime as LOCAL time, which on a
 * UTC+7 machine shifted every marker ~7h off the candles. The parquet `time`
 * column, by contrast, decodes to a correct UTC epoch — hence the constant
 * offset between candles and markers. Force UTC interpretation here.
 */
function parseTradeTimestamp(raw: string | undefined): number | null {
  if (raw === undefined) return null;
  const s = raw.trim();
  if (s === '') return null;

  // Already epoch (number) — delegate to the shared numeric path.
  const asNum = Number(s);
  if (Number.isFinite(asNum) && s !== '') return toTimestamp(asNum);

  // "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS" with optional fractional
  // seconds, no timezone -> treat as UTC.
  const m = s.match(
    /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?$/,
  );
  if (m) {
    const [, y, mo, d, h, mi, se, frac] = m;
    const ms = frac ? Number(`0.${frac}`) * 1000 : 0;
    return Date.UTC(+y, +mo - 1, +d, +h, +mi, +se, ms);
  }

  // Has an explicit timezone (Z or +hh:mm) or other ISO form — Date.parse is
  // unambiguous there.
  const parsed = Date.parse(s);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * Parse a trades.csv produced by the trainers or run_baselines.
 * Trainer header:  episode,phase,day,entry_time,exit_time,side,entry_price,exit_price,bars_held,pnl_log
 * Baseline header: baseline,phase,day,entry_time,exit_time,side,entry_price,exit_price,bars_held,pnl_log
 */
export function parseTradesCsv(text: string): Trade[] {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== '');
  if (lines.length < 2) return [];
  const header = splitCsvLine(lines[0]).map((h) => h.trim().toLowerCase());
  const col = (name: string) => header.indexOf(name);

  const groupIdx = col('episode') >= 0 ? col('episode') : col('baseline');
  const phaseIdx = col('phase');
  const dayIdx = col('day');
  const entryTimeIdx = col('entry_time');
  const exitTimeIdx = col('exit_time');
  const sideIdx = col('side');
  const entryPriceIdx = col('entry_price');
  const exitPriceIdx = col('exit_price');
  const barsHeldIdx = col('bars_held');
  const pnlIdx = col('pnl_log');

  if (entryTimeIdx < 0 || exitTimeIdx < 0 || sideIdx < 0) {
    throw new Error('trades.csv missing entry_time/exit_time/side columns.');
  }

  const trades: Trade[] = [];
  for (let i = 1; i < lines.length; i += 1) {
    const f = splitCsvLine(lines[i]);
    const entryTime = parseTradeTimestamp(f[entryTimeIdx]);
    const exitTime = parseTradeTimestamp(f[exitTimeIdx]);
    if (entryTime === null || exitTime === null) continue;
    const sideRaw = (f[sideIdx] ?? '').trim().toLowerCase();
    const side: 'long' | 'short' = sideRaw === 'short' || sideRaw === '-1' ? 'short' : 'long';
    trades.push({
      group: groupIdx >= 0 ? (f[groupIdx] ?? '').trim() : '',
      phase: phaseIdx >= 0 ? (f[phaseIdx] ?? '').trim() : '',
      day: dayIdx >= 0 ? (f[dayIdx] ?? '').trim() : '',
      entryTime,
      exitTime,
      side,
      entryPrice: entryPriceIdx >= 0 ? toNumeric(f[entryPriceIdx]) ?? NaN : NaN,
      exitPrice: exitPriceIdx >= 0 ? toNumeric(f[exitPriceIdx]) ?? NaN : NaN,
      barsHeld: barsHeldIdx >= 0 ? toNumeric(f[barsHeldIdx]) ?? 0 : 0,
      pnlLog: pnlIdx >= 0 ? toNumeric(f[pnlIdx]) ?? 0 : 0,
    });
  }
  return trades;
}

export async function loadTradesCsv(file: File): Promise<Trade[]> {
  const text = await file.text();
  const trades = parseTradesCsv(text);
  if (trades.length === 0) throw new Error('No trades parsed from CSV.');
  return trades;
}

// ---------------------------------------------------------------------------
// Filtering helpers
// ---------------------------------------------------------------------------

export function uniquePhases(trades: Trade[]): string[] {
  return Array.from(new Set(trades.map((t) => t.phase))).filter(Boolean).sort();
}

export function uniqueEpisodes(trades: Trade[], phase: string): string[] {
  return Array.from(
    new Set(trades.filter((t) => phase === '' || t.phase === phase).map((t) => t.group)),
  )
    .filter(Boolean)
    .sort((a, b) => {
      const na = Number(a);
      const nb = Number(b);
      if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
      return a.localeCompare(b);
    });
}

export function filterTrades(trades: Trade[], phase: string, episode: string): Trade[] {
  return trades.filter(
    (t) => (phase === '' || t.phase === phase) && (episode === '' || t.group === episode),
  );
}

/**
 * Human-readable time span covered by the currently filtered trades, e.g.
 * "2026-01-15 01:03 -> 2026-04-30 23:58 (1,842 trades)". Returns null when
 * there are no trades in the selection.
 */
export function tradesTimeRangeLabel(trades: Trade[]): string | null {
  if (trades.length === 0) return null;
  let lo = Infinity;
  let hi = -Infinity;
  for (const t of trades) {
    lo = Math.min(lo, t.entryTime, t.exitTime);
    hi = Math.max(hi, t.entryTime, t.exitTime);
  }
  const fmt = (ms: number) => {
    const d = new Date(ms);
    const p = (n: number) => String(n).padStart(2, '0');
    return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ` +
      `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}`;
  };
  return `${fmt(lo)} -> ${fmt(hi)} (${trades.length.toLocaleString()} trades)`;
}

/** Largest episode id (numeric if possible) for a phase — used as the default
 * selection so the chart opens on the most recent episode, not "All". */
export function lastEpisode(trades: Trade[], phase: string): string {
  const eps = uniqueEpisodes(trades, phase);
  return eps.length > 0 ? eps[eps.length - 1] : '';
}

/**
 * Snap a trade timestamp to the start time of the actual candle that contains
 * it. Binary-searches the rendered `bars` (which are already sorted ascending
 * by `t`) for the last bar whose start <= ts, then verifies ts is within that
 * bar's bucket. Returns the bar's `t` so the marker x lands exactly on a drawn
 * candle (fixes markers floating between candles), or null if ts is outside
 * the chart window.
 *
 * This replaces the old bucket-arithmetic snap, which drifted because the GOLD
 * session does not start on a timeframe boundary (e.g. ~01:00, not 00:00).
 */
export function snapTsToBar(ts: number, bars: Bar[], tf: Timeframe): number | null {
  if (bars.length === 0) return null;
  if (ts < bars[0].t) return null;
  const bucketMs = TF_MINUTES[tf] * 60_000;

  let lo = 0;
  let hi = bars.length - 1;
  let idx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (bars[mid].t <= ts) {
      idx = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (idx < 0) return null;
  // Reject if ts falls past this bar's bucket and there is a gap before the
  // next bar (e.g. trade timestamp lands in a session gap).
  const barStart = bars[idx].t;
  if (ts - barStart >= bucketMs && idx + 1 < bars.length && bars[idx + 1].t > ts) {
    // ts is inside a gap; still snap to the nearest preceding bar.
    return barStart;
  }
  return barStart;
}

/** Hard cap on rendered candles. Plotly candlestick is SVG (not WebGL); past
 * a few thousand shapes the browser tab freezes. The full GOLD_M1 parquet is
 * ~1.03M rows, so the chart must always be clipped/capped before rendering. */
export const MAX_CANDLES = 4000;

/**
 * Restrict bars to the time window spanned by the shown trades (plus padding),
 * then hard-cap the count. Without this, uploading the full GOLD_M1 parquet and
 * picking M1 would try to draw ~1M candles and hang the tab.
 *
 * Returns `{ bars, clipped, capped }` so the UI can explain what it did.
 */
export function clipBarsToTrades(
  bars: Bar[],
  trades: Trade[],
  tf: Timeframe,
): { bars: Bar[]; clipped: boolean; capped: boolean } {
  if (bars.length === 0) return { bars, clipped: false, capped: false };

  let lo: number;
  let hi: number;
  if (trades.length === 0) {
    // No trades selected: show the most recent window only.
    hi = bars[bars.length - 1].t;
    lo = bars[0].t;
  } else {
    lo = Infinity;
    hi = -Infinity;
    for (const t of trades) {
      lo = Math.min(lo, t.entryTime, t.exitTime);
      hi = Math.max(hi, t.entryTime, t.exitTime);
    }
    // Pad by ~20 buckets each side so trades aren't glued to the chart edge.
    const padMs = TF_MINUTES[tf] * 60_000 * 20;
    lo -= padMs;
    hi += padMs;
  }

  let windowBars = bars.filter((b) => b.t >= lo && b.t <= hi);
  const clipped = windowBars.length < bars.length;

  let capped = false;
  if (windowBars.length > MAX_CANDLES) {
    // Keep the most recent MAX_CANDLES bars in the window.
    windowBars = windowBars.slice(windowBars.length - MAX_CANDLES);
    capped = true;
  }
  return { bars: windowBars, clipped, capped };
}
