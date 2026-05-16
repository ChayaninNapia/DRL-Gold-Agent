import { parquetMetadataAsync, parquetReadObjects, parquetSchema } from 'hyparquet';
import { compressors } from 'hyparquet-compressors';
import type { LoadedParquet } from '../types/parquet';
import { inferColumns, normalizeRows } from './dataUtils';

interface BrowserAsyncBuffer {
  byteLength: number;
  slice(start: number, end?: number): Promise<ArrayBuffer>;
}

function asyncBufferFromBrowserFile(file: File): BrowserAsyncBuffer {
  return {
    byteLength: file.size,
    slice: (start: number, end?: number) => file.slice(start, end).arrayBuffer(),
  };
}

export async function readParquetFile(file: File): Promise<LoadedParquet> {
  const asyncFile = asyncBufferFromBrowserFile(file);
  const metadata = await parquetMetadataAsync(asyncFile);
  const schema = parquetSchema(metadata);
  const schemaColumns = schema.children.map((child) => child.element.name);
  const rawRows = await parquetReadObjects({ file: asyncFile, compressors });
  const rows = normalizeRows(rawRows);
  const columns = schemaColumns.length > 0 ? schemaColumns : inferColumns(rows);

  return {
    fileName: file.name,
    fileSize: file.size,
    rowCount: Number(metadata.num_rows ?? rows.length),
    columns,
    rows,
  };
}
