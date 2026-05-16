import { useMemo, useState } from 'react';
import ColumnPanel from './components/ColumnPanel';
import DataTable from './components/DataTable';
import FileUpload from './components/FileUpload';
import LinePlot from './components/LinePlot';
import ScatterPlot from './components/ScatterPlot';
import SelfTests from './components/SelfTests';
import SummaryCards from './components/SummaryCards';
import { defaultLineY, defaultScatterX, defaultScatterY } from './lib/chartUtils';
import { findDefaultTimeColumn, getNumericColumns, inferSchema, missingExpectedColumns } from './lib/dataUtils';
import { readParquetFile } from './lib/parquet';
import type { LoadedParquet } from './types/parquet';

export default function ParquetView() {
  const [loaded, setLoaded] = useState<LoadedParquet | null>(null);
  const [visibleColumns, setVisibleColumns] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const schema = useMemo(() => inferSchema(loaded?.rows ?? [], loaded?.columns ?? []), [loaded]);
  const numericColumns = useMemo(() => getNumericColumns(loaded?.rows ?? [], loaded?.columns ?? []), [loaded]);
  const missingColumns = useMemo(() => missingExpectedColumns(loaded?.columns ?? []), [loaded]);
  const timeColumn = findDefaultTimeColumn(loaded?.columns ?? []);
  const lineY = defaultLineY(loaded?.columns ?? [], numericColumns);
  const scatterX = defaultScatterX(numericColumns);
  const scatterY = defaultScatterY(numericColumns);

  async function handleFileSelected(file: File) {
    setLoading(true);
    setError(null);
    try {
      const result = await readParquetFile(file);
      setLoaded(result);
      setVisibleColumns(result.columns);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(`Could not read parquet file. ${message}`);
      setLoaded(null);
      setVisibleColumns([]);
    } finally {
      setLoading(false);
    }
  }

  function toggleColumn(column: string) {
    setVisibleColumns((current) => (current.includes(column) ? current.filter((item) => item !== column) : [...current, column]));
  }

  return (
    <div className="flex flex-col gap-4">
      <FileUpload loading={loading} error={error} onFileSelected={handleFileSelected} />

      <SummaryCards fileName={loaded?.fileName} fileSize={loaded?.fileSize} rowCount={loaded?.rowCount} columns={loaded?.columns ?? []} missingColumns={missingColumns} />

      {!loaded ? (
        <SelfTests />
      ) : (
        <div className="grid gap-4 xl:grid-cols-[360px_1fr]">
          <ColumnPanel schema={schema} visibleColumns={visibleColumns} onToggleColumn={toggleColumn} onSelectAll={() => setVisibleColumns(loaded.columns)} onClear={() => setVisibleColumns([])} />
          <div className="flex min-w-0 flex-col gap-4">
            <DataTable rows={loaded.rows} visibleColumns={visibleColumns} />
            {numericColumns.length > 0 && timeColumn ? (
              <LinePlot key={`${loaded.fileName}-line`} rows={loaded.rows} columns={loaded.columns} numericColumns={numericColumns} defaultTimeColumn={timeColumn} defaultYColumn={lineY} />
            ) : (
              <section className="panel p-4 text-sm text-slate-400">Line plot needs one time-like column and one numeric column.</section>
            )}
            {numericColumns.length > 1 ? (
              <ScatterPlot key={`${loaded.fileName}-scatter`} rows={loaded.rows} numericColumns={numericColumns} defaultXColumn={scatterX} defaultYColumn={scatterY} />
            ) : (
              <section className="panel p-4 text-sm text-slate-400">Scatter plot needs at least two numeric columns.</section>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
