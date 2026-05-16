export type ParquetRow = Record<string, unknown>;

export type ColumnKind = 'time' | 'numeric' | 'text' | 'unknown';

export interface ColumnInfo {
  name: string;
  kind: ColumnKind;
  sample: unknown;
}

export interface LoadedParquet {
  fileName: string;
  fileSize: number;
  rowCount: number;
  columns: string[];
  rows: ParquetRow[];
}
