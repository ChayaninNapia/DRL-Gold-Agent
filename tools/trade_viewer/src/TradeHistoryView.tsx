import { useMemo, useRef, useState } from 'react';
import TradeChart from './components/TradeChart';
import {
  TIMEFRAMES,
  clipBarsToTrades,
  filterTrades,
  lastEpisode,
  loadOhlcParquet,
  loadTradesCsv,
  resampleBars,
  tradesTimeRangeLabel,
  uniqueEpisodes,
  uniquePhases,
  type Bar,
  type Timeframe,
  type Trade,
} from './lib/tradeData';

export default function TradeHistoryView() {
  const [m1, setM1] = useState<Bar[] | null>(null);
  const [ohlcName, setOhlcName] = useState<string | null>(null);
  const [trades, setTrades] = useState<Trade[] | null>(null);
  const [tradesName, setTradesName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingOhlc, setLoadingOhlc] = useState(false);
  const [loadingTrades, setLoadingTrades] = useState(false);

  const [timeframe, setTimeframe] = useState<Timeframe>('M1');
  const [phase, setPhase] = useState('');
  const [episode, setEpisode] = useState('');

  const ohlcRef = useRef<HTMLInputElement | null>(null);
  const tradesRef = useRef<HTMLInputElement | null>(null);

  const phases = useMemo(() => (trades ? uniquePhases(trades) : []), [trades]);
  const episodes = useMemo(() => (trades ? uniqueEpisodes(trades, phase) : []), [trades, phase]);

  const shownTrades = useMemo(
    () => (trades ? filterTrades(trades, phase, episode) : []),
    [trades, phase, episode],
  );
  const rangeLabel = useMemo(() => tradesTimeRangeLabel(shownTrades), [shownTrades]);
  const episodeValid = episode === '' || episodes.includes(episode);
  const chart = useMemo(() => {
    if (!m1) return { bars: [] as Bar[], clipped: false, capped: false };
    const resampled = resampleBars(m1, timeframe);
    return clipBarsToTrades(resampled, shownTrades, timeframe);
  }, [m1, timeframe, shownTrades]);
  const bars = chart.bars;

  async function handleOhlc(file: File) {
    setLoadingOhlc(true);
    setError(null);
    try {
      const { loaded, m1: bars } = await loadOhlcParquet(file);
      setM1(bars);
      setOhlcName(`${loaded.fileName} (${bars.length.toLocaleString()} M1 bars)`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setM1(null);
      setOhlcName(null);
    } finally {
      setLoadingOhlc(false);
    }
  }

  async function handleTrades(file: File) {
    setLoadingTrades(true);
    setError(null);
    try {
      const parsed = await loadTradesCsv(file);
      setTrades(parsed);
      setTradesName(`${file.name} (${parsed.length.toLocaleString()} trades)`);
      const ph = uniquePhases(parsed);
      // Prefer test for final-result inspection, else first phase.
      const defaultPhase = ph.includes('test') ? 'test' : ph[0] ?? '';
      setPhase(defaultPhase);
      // Default to the LAST episode (not "All") — "All" renders every trade
      // across the run and is slow on large CSVs.
      setEpisode(lastEpisode(parsed, defaultPhase));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setTrades(null);
      setTradesName(null);
    } finally {
      setLoadingTrades(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <section className="panel p-5">
        <h1 className="text-2xl font-semibold text-slate-50">Trade History</h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload the OHLC parquet (e.g. GOLD_M1) and a trades.csv from a run. Candles come from
          the parquet; entry/exit markers come from the CSV.
        </p>

        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <div>
            <span className="label">OHLC parquet</span>
            <input
              ref={ohlcRef}
              className="hidden"
              type="file"
              accept=".parquet"
              onChange={(e) => {
                const f = e.currentTarget.files?.[0];
                if (f) handleOhlc(f);
              }}
            />
            <button
              className="btn btn-primary mt-1 w-full"
              type="button"
              disabled={loadingOhlc}
              onClick={() => ohlcRef.current?.click()}
            >
              {loadingOhlc ? 'Loading...' : ohlcName ?? 'Upload OHLC .parquet'}
            </button>
          </div>
          <div>
            <span className="label">Trades CSV</span>
            <input
              ref={tradesRef}
              className="hidden"
              type="file"
              accept=".csv"
              onChange={(e) => {
                const f = e.currentTarget.files?.[0];
                if (f) handleTrades(f);
              }}
            />
            <button
              className="btn btn-primary mt-1 w-full"
              type="button"
              disabled={loadingTrades}
              onClick={() => tradesRef.current?.click()}
            >
              {loadingTrades ? 'Loading...' : tradesName ?? 'Upload trades.csv'}
            </button>
          </div>
        </div>

        {error ? (
          <div className="mt-4 rounded-md border border-red-900 bg-red-950 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        ) : null}

        {trades ? (
          <>
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <label className="text-sm">
              <span className="label">Timeframe</span>
              <select
                className="control mt-1 w-full"
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value as Timeframe)}
              >
                {TIMEFRAMES.map((tf) => (
                  <option key={tf} value={tf}>
                    {tf}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-sm">
              <span className="label">Phase</span>
              <select
                className="control mt-1 w-full"
                value={phase}
                onChange={(e) => {
                  const p = e.target.value;
                  setPhase(p);
                  // Default to the last episode of the new phase, not "All".
                  setEpisode(trades ? lastEpisode(trades, p) : '');
                }}
              >
                <option value="">All</option>
                {phases.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-sm">
              <span className="label">Episode</span>
              <input
                className="control mt-1 w-full"
                type="text"
                inputMode="numeric"
                list="episode-list"
                placeholder="All"
                value={episode}
                onChange={(e) => setEpisode(e.target.value.trim())}
              />
              <datalist id="episode-list">
                {episodes.map((ep) => (
                  <option key={ep} value={ep} />
                ))}
              </datalist>
            </label>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
            <span className="text-slate-400">
              {episodes.length.toLocaleString()} episodes in phase
              {phase ? ` "${phase}"` : ''} (e.g. {episodes[0] ?? '-'} ..{' '}
              {episodes[episodes.length - 1] ?? '-'})
            </span>
            {!episodeValid ? (
              <span className="text-amber-300">
                episode "{episode}" not found in this phase — showing nothing
              </span>
            ) : rangeLabel ? (
              <span className="text-slate-300">data: {rangeLabel}</span>
            ) : (
              <span className="text-slate-500">no trades in selection</span>
            )}
          </div>
          </>
        ) : null}
      </section>

      {m1 && trades ? (
        <>
          {chart.clipped || chart.capped ? (
            <div className="rounded-md border border-amber-700 bg-amber-950 px-4 py-2 text-xs text-amber-200">
              {chart.capped
                ? `Showing the most recent ${bars.length.toLocaleString()} candles around the selected trades (chart is capped to keep the browser responsive). Narrow the phase/episode filter or pick a coarser timeframe to see a different window.`
                : `Chart clipped to the time window of the selected trades (${bars.length.toLocaleString()} candles).`}
            </div>
          ) : null}
          <TradeChart bars={bars} trades={shownTrades} timeframe={timeframe} />
        </>
      ) : (
        <section className="panel p-6 text-sm text-slate-400">
          Upload both an OHLC parquet and a trades.csv to see the candlestick chart with trade
          markers.
        </section>
      )}
    </div>
  );
}
