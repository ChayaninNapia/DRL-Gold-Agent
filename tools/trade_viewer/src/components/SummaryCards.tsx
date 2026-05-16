import { formatBytes } from '../lib/formatting';

interface SummaryCardsProps {
  fileName?: string;
  fileSize?: number;
  rowCount?: number;
  columns: string[];
  missingColumns: string[];
}

export default function SummaryCards({ fileName, fileSize, rowCount, columns, missingColumns }: SummaryCardsProps) {
  const cards = [
    { label: 'File', value: fileName ?? 'No file loaded' },
    { label: 'Size', value: fileSize === undefined ? '-' : formatBytes(fileSize) },
    { label: 'Rows', value: rowCount === undefined ? '-' : rowCount.toLocaleString() },
    { label: 'Columns', value: columns.length.toLocaleString() },
  ];

  return (
    <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {cards.map((card) => (
        <div className="panel p-4" key={card.label}>
          <div className="label">{card.label}</div>
          <div className="mt-2 truncate text-lg font-semibold text-slate-50" title={String(card.value)}>
            {card.value}
          </div>
        </div>
      ))}
      {missingColumns.length > 0 ? (
        <div className="panel border-amber-700 bg-amber-950 p-4 sm:col-span-2 xl:col-span-4">
          <div className="label text-amber-300">MT5 column warning</div>
          <p className="mt-2 text-sm text-amber-200">Missing expected columns: {missingColumns.join(', ')}</p>
        </div>
      ) : null}
    </section>
  );
}
