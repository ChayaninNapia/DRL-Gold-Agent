import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import { downsampleRows, filterRowsByTime, getExtentByAccessor } from '../lib/chartUtils';
import { datetimeLocalToTimestamp, toDatetimeLocalValue, toNumeric, toTimestamp, valueToDisplay } from '../lib/formatting';
import type { ParquetRow } from '../types/parquet';

interface LinePlotProps {
  rows: ParquetRow[];
  columns: string[];
  numericColumns: string[];
  defaultTimeColumn: string;
  defaultYColumn: string;
  maxPoints?: number;
}

export default function LinePlot({ rows, columns, numericColumns, defaultTimeColumn, defaultYColumn, maxPoints = 5000 }: LinePlotProps) {
  const [timeColumn, setTimeColumn] = useState(defaultTimeColumn);
  const [yColumn, setYColumn] = useState(defaultYColumn);

  const timeExtent = useMemo(() => getExtentByAccessor(rows, (row) => toTimestamp(row[timeColumn])), [rows, timeColumn]);
  const [startValue, setStartValue] = useState('');
  const [endValue, setEndValue] = useState('');

  const chartRows = useMemo(() => {
    const start = datetimeLocalToTimestamp(startValue);
    const end = datetimeLocalToTimestamp(endValue);
    const filtered = filterRowsByTime(rows, timeColumn, start, end, toTimestamp).filter((row) => toTimestamp(row[timeColumn]) !== null && toNumeric(row[yColumn]) !== null);
    return {
      rawCount: filtered.length,
      sampled: downsampleRows(filtered, maxPoints),
    };
  }, [rows, timeColumn, yColumn, startValue, endValue, maxPoints]);

  const x = chartRows.sampled.map((row) => new Date(toTimestamp(row[timeColumn]) ?? 0));
  const y = chartRows.sampled.map((row) => toNumeric(row[yColumn]) ?? 0);

  return (
    <section className="panel overflow-hidden">
      <div className="border-b border-slate-700 px-4 py-3">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
          <div>
            <h2 className="font-semibold text-slate-50">Line Plot vs Time</h2>
            <p className="text-xs text-slate-400">
              Plotting {chartRows.sampled.length.toLocaleString()} of {chartRows.rawCount.toLocaleString()} selected points
            </p>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-4">
            <label className="text-sm">
              <span className="label">Time</span>
              <select className="control mt-1 w-full" value={timeColumn} onChange={(event) => setTimeColumn(event.target.value)}>
                {columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-sm">
              <span className="label">Y</span>
              <select className="control mt-1 w-full" value={yColumn} onChange={(event) => setYColumn(event.target.value)}>
                {numericColumns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-sm">
              <span className="label">Start</span>
              <input className="control mt-1 w-full" type="datetime-local" value={startValue} placeholder={toDatetimeLocalValue(timeExtent?.[0])} onChange={(event) => setStartValue(event.target.value)} />
            </label>
            <label className="text-sm">
              <span className="label">End</span>
              <input className="control mt-1 w-full" type="datetime-local" value={endValue} placeholder={toDatetimeLocalValue(timeExtent?.[1])} onChange={(event) => setEndValue(event.target.value)} />
            </label>
          </div>
        </div>
      </div>
      <div className="h-[420px]">
        <Plot
          data={[
            {
              type: 'scattergl',
              mode: 'lines',
              x,
              y,
              line: { color: '#22d3ee', width: 1.4 },
              hovertemplate: `%{x}<br>${yColumn}: %{y:.2f}<extra></extra>`,
            },
          ]}
          layout={{
            autosize: true,
            margin: { l: 58, r: 20, t: 20, b: 48 },
            xaxis: {
              title: timeColumn,
              rangeslider: { visible: false },
              gridcolor: '#1e293b',
              linecolor: '#334155',
              tickfont: { color: '#94a3b8' },
            },
            yaxis: {
              title: yColumn,
              tickformat: ',.2f',
              gridcolor: '#1e293b',
              linecolor: '#334155',
              tickfont: { color: '#94a3b8' },
            },
            font: { color: '#cbd5e1' },
            paper_bgcolor: '#0f172a',
            plot_bgcolor: '#0b1120',
          }}
          config={{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
      {chartRows.rawCount === 0 ? <div className="px-4 pb-4 text-sm text-slate-400">No numeric points found for {valueToDisplay(yColumn)}.</div> : null}
    </section>
  );
}
