import type { ColumnInfo } from '../types/parquet';
import { valueToDisplay } from '../lib/formatting';

interface ColumnPanelProps {
  schema: ColumnInfo[];
  visibleColumns: string[];
  onToggleColumn: (column: string) => void;
  onSelectAll: () => void;
  onClear: () => void;
}

const kindClass: Record<ColumnInfo['kind'], string> = {
  time: 'bg-cyan-950 text-cyan-200 border-cyan-700',
  numeric: 'bg-emerald-950 text-emerald-200 border-emerald-700',
  text: 'bg-slate-800 text-slate-200 border-slate-600',
  unknown: 'bg-amber-950 text-amber-200 border-amber-700',
};

export default function ColumnPanel({ schema, visibleColumns, onToggleColumn, onSelectAll, onClear }: ColumnPanelProps) {
  const visible = new Set(visibleColumns);

  return (
    <section className="panel overflow-hidden">
      <div className="flex items-center justify-between border-b border-slate-700 px-4 py-3">
        <div>
          <h2 className="font-semibold text-slate-50">Schema / Columns</h2>
          <p className="text-xs text-slate-400">{visibleColumns.length} visible</p>
        </div>
        <div className="flex gap-2">
          <button className="btn px-2 py-1 text-xs" type="button" onClick={onSelectAll}>
            Select
          </button>
          <button className="btn px-2 py-1 text-xs" type="button" onClick={onClear}>
            Clear
          </button>
        </div>
      </div>
      <div className="max-h-[430px] overflow-auto">
        {schema.map((column) => (
          <label className="grid cursor-pointer grid-cols-[auto_1fr] gap-3 border-b border-slate-800 px-4 py-3 hover:bg-slate-800" key={column.name}>
            <input checked={visible.has(column.name)} className="mt-1 h-4 w-4 accent-cyan-500" type="checkbox" onChange={() => onToggleColumn(column.name)} />
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-medium text-slate-100">{column.name}</span>
                <span className={`rounded border px-1.5 py-0.5 text-[11px] font-medium ${kindClass[column.kind]}`}>{column.kind}</span>
              </div>
              <div className="mt-1 truncate text-xs text-slate-400" title={valueToDisplay(column.sample)}>
                Sample: {valueToDisplay(column.sample) || '-'}
              </div>
            </div>
          </label>
        ))}
      </div>
    </section>
  );
}
