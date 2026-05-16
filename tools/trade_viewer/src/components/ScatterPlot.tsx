import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import { downsampleRows } from '../lib/chartUtils';
import { toNumeric } from '../lib/formatting';
import type { ParquetRow } from '../types/parquet';

interface ScatterPlotProps {
  rows: ParquetRow[];
  numericColumns: string[];
  defaultXColumn: string;
  defaultYColumn: string;
  maxPoints?: number;
}

export default function ScatterPlot({ rows, numericColumns, defaultXColumn, defaultYColumn, maxPoints = 5000 }: ScatterPlotProps) {
  const [xColumn, setXColumn] = useState(defaultXColumn);
  const [yColumn, setYColumn] = useState(defaultYColumn);

  const chartRows = useMemo(() => {
    const filtered = rows.filter((row) => toNumeric(row[xColumn]) !== null && toNumeric(row[yColumn]) !== null);
    return {
      rawCount: filtered.length,
      sampled: downsampleRows(filtered, maxPoints),
    };
  }, [rows, xColumn, yColumn, maxPoints]);

  const x = chartRows.sampled.map((row) => toNumeric(row[xColumn]) ?? 0);
  const y = chartRows.sampled.map((row) => toNumeric(row[yColumn]) ?? 0);

  return (
    <section className="panel overflow-hidden">
      <div className="flex flex-col gap-3 border-b border-slate-700 px-4 py-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="font-semibold text-slate-50">Scatter Plot</h2>
          <p className="text-xs text-slate-400">
            Plotting {chartRows.sampled.length.toLocaleString()} of {chartRows.rawCount.toLocaleString()} selected points
          </p>
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          <label className="text-sm">
            <span className="label">X</span>
            <select className="control mt-1 w-full" value={xColumn} onChange={(event) => setXColumn(event.target.value)}>
              {numericColumns.map((column) => (
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
        </div>
      </div>
      <div className="h-[420px]">
        <Plot
          data={[
            {
              type: 'scattergl',
              mode: 'markers',
              x,
              y,
              marker: { color: '#38bdf8', size: 5, opacity: 0.7 },
              hovertemplate: `${xColumn}: %{x:.2f}<br>${yColumn}: %{y:.2f}<extra></extra>`,
            },
          ]}
          layout={{
            autosize: true,
            dragmode: 'zoom',
            margin: { l: 58, r: 20, t: 20, b: 50 },
            xaxis: {
              title: xColumn,
              tickformat: ',.2f',
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
    </section>
  );
}
