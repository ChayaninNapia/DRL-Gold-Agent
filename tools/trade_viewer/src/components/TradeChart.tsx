import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import type { Data } from 'plotly.js';
import type { Bar, Trade, Timeframe } from '../lib/tradeData';
import { snapTsToBar } from '../lib/tradeData';

interface TradeChartProps {
  bars: Bar[];
  trades: Trade[];
  timeframe: Timeframe;
}

const UP = '#26a69a';
const DOWN = '#ef5350';
const LONG = '#3b82f6'; // blue: long entry/exit
const SHORT = '#ef4444'; // red: short entry/exit

// Format an epoch (ms) as a naive "YYYY-MM-DD HH:MM:SS" string in UTC.
// Passing naive strings (no Z/offset) to Plotly makes it plot the wall-clock
// time as-is without applying the browser's local timezone. Both candles and
// markers use this, so the x-axis reads the same UTC time the trainers wrote
// into trades.csv and the parquet — no local-timezone shift on the axis.
function utcAxis(ms: number): string {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, '0');
  return (
    `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ` +
    `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}`
  );
}

export default function TradeChart({ bars, trades, timeframe }: TradeChartProps) {
  // uirevision keeps zoom/pan across re-renders; bumping resetKey forces a
  // fresh autorange (Reset View button).
  const [resetKey, setResetKey] = useState(0);

  const x = useMemo(() => bars.map((b) => utcAxis(b.t)), [bars]);
  const open = useMemo(() => bars.map((b) => b.o), [bars]);
  const high = useMemo(() => bars.map((b) => b.h), [bars]);
  const low = useMemo(() => bars.map((b) => b.l), [bars]);
  const close = useMemo(() => bars.map((b) => b.c), [bars]);

  // Snap markers to the candle that contains the trade timestamp.
  const markers = useMemo(() => {
    const longEntryX: string[] = [];
    const longEntryY: number[] = [];
    const shortEntryX: string[] = [];
    const shortEntryY: number[] = [];
    const longExitX: string[] = [];
    const longExitY: number[] = [];
    const shortExitX: string[] = [];
    const shortExitY: number[] = [];
    for (const t of trades) {
      const tIn = snapTsToBar(t.entryTime, bars, timeframe);
      const tOut = snapTsToBar(t.exitTime, bars, timeframe);
      if (t.side === 'long') {
        if (tIn !== null) {
          longEntryX.push(utcAxis(tIn));
          longEntryY.push(t.entryPrice);
        }
        if (tOut !== null) {
          longExitX.push(utcAxis(tOut));
          longExitY.push(t.exitPrice);
        }
      } else {
        if (tIn !== null) {
          shortEntryX.push(utcAxis(tIn));
          shortEntryY.push(t.entryPrice);
        }
        if (tOut !== null) {
          shortExitX.push(utcAxis(tOut));
          shortExitY.push(t.exitPrice);
        }
      }
    }
    return {
      longEntryX, longEntryY, shortEntryX, shortEntryY,
      longExitX, longExitY, shortExitX, shortExitY,
    };
  }, [trades, timeframe, bars]);

  // @types/plotly.js for this version omits candlestick fillcolor, but Plotly
  // supports it at runtime — cast just this trace.
  const candleTrace = {
    type: 'candlestick',
    x,
    open,
    high,
    low,
    close,
    name: 'Price',
    increasing: { line: { color: UP }, fillcolor: UP },
    decreasing: { line: { color: DOWN }, fillcolor: DOWN },
    hoverlabel: { bgcolor: '#0b0f1a' },
  } as unknown as Data;

  return (
    <section className="panel overflow-hidden">
      <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
        <div>
          <h2 className="font-semibold text-slate-50">Price &amp; Trades</h2>
          <p className="text-xs text-slate-500">
            {bars.length.toLocaleString()} {timeframe} candles &middot; {trades.length.toLocaleString()} trades
            &middot; scroll to zoom, drag to pan, double-click to reset
          </p>
        </div>
        <button className="btn" type="button" onClick={() => setResetKey((k) => k + 1)}>
          Reset View
        </button>
      </div>
      <div className="h-[560px]">
        <Plot
          key={resetKey}
          data={[
            candleTrace,
            {
              type: 'scatter',
              mode: 'markers',
              name: 'Long entry',
              x: markers.longEntryX,
              y: markers.longEntryY,
              marker: { symbol: 'triangle-up', size: 13, color: LONG, line: { color: '#bfdbfe', width: 1 } },
              hovertemplate: 'LONG entry<br>%{x}<br>%{y:.2f}<extra></extra>',
            },
            {
              type: 'scatter',
              mode: 'markers',
              name: 'Short entry',
              x: markers.shortEntryX,
              y: markers.shortEntryY,
              marker: { symbol: 'triangle-down', size: 13, color: SHORT, line: { color: '#fecaca', width: 1 } },
              hovertemplate: 'SHORT entry<br>%{x}<br>%{y:.2f}<extra></extra>',
            },
            {
              type: 'scatter',
              mode: 'markers',
              name: 'Long exit',
              x: markers.longExitX,
              y: markers.longExitY,
              marker: { symbol: 'x', size: 11, color: LONG, line: { color: '#bfdbfe', width: 1 } },
              hovertemplate: 'LONG exit<br>%{x}<br>%{y:.2f}<extra></extra>',
            },
            {
              type: 'scatter',
              mode: 'markers',
              name: 'Short exit',
              x: markers.shortExitX,
              y: markers.shortExitY,
              marker: { symbol: 'x', size: 11, color: SHORT, line: { color: '#fecaca', width: 1 } },
              hovertemplate: 'SHORT exit<br>%{x}<br>%{y:.2f}<extra></extra>',
            },
          ]}
          layout={{
            autosize: true,
            dragmode: 'pan',
            uirevision: `tf-${timeframe}-${resetKey}`,
            margin: { l: 62, r: 16, t: 12, b: 44 },
            legend: { orientation: 'h', y: 1.04, font: { color: '#cbd5e1' } },
            xaxis: {
              rangeslider: { visible: false },
              gridcolor: '#1e293b',
              linecolor: '#334155',
              tickfont: { color: '#94a3b8' },
              type: 'date',
            },
            yaxis: {
              gridcolor: '#1e293b',
              linecolor: '#334155',
              tickfont: { color: '#94a3b8' },
              tickformat: ',.2f',
              fixedrange: false,
            },
            paper_bgcolor: '#0f172a',
            plot_bgcolor: '#0b1120',
          }}
          config={{
            responsive: true,
            displaylogo: false,
            scrollZoom: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </section>
  );
}
