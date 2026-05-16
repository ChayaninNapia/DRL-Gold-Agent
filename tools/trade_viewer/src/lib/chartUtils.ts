import type { ParquetRow } from '../types/parquet';

export function getExtent(values: ArrayLike<number>): [number, number] | null {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let found = false;

  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
    found = true;
  }

  return found ? [min, max] : null;
}

export function getExtentByAccessor<T>(rows: T[], accessor: (row: T) => number | null): [number, number] | null {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let found = false;

  for (const row of rows) {
    const value = accessor(row);
    if (value === null || !Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
    found = true;
  }

  return found ? [min, max] : null;
}

export function downsampleRows<T>(rows: T[], maxPoints = 5000): T[] {
  if (maxPoints < 2) return rows.slice(0, Math.max(0, maxPoints));
  if (rows.length <= maxPoints) return rows;

  const sampled: T[] = [];
  const lastIndex = rows.length - 1;
  const step = lastIndex / (maxPoints - 1);
  let previousIndex = -1;

  for (let point = 0; point < maxPoints; point += 1) {
    const index = point === maxPoints - 1 ? lastIndex : Math.round(point * step);
    if (index !== previousIndex) {
      sampled.push(rows[index]);
      previousIndex = index;
    }
  }

  return sampled;
}

export function defaultLineY(columns: string[], numericColumns: string[]): string {
  return numericColumns.find((column) => column.toLowerCase() === 'close') ?? numericColumns[0] ?? columns[0] ?? '';
}

export function defaultScatterX(numericColumns: string[]): string {
  return numericColumns.find((column) => column.toLowerCase() === 'tick_volume') ?? numericColumns[0] ?? '';
}

export function defaultScatterY(numericColumns: string[]): string {
  return (
    numericColumns.find((column) => column.toLowerCase() === 'spread') ??
    numericColumns.find((column) => column.toLowerCase() === 'close') ??
    numericColumns.find((column) => column !== defaultScatterX(numericColumns)) ??
    numericColumns[0] ??
    ''
  );
}

export function filterRowsByTime(rows: ParquetRow[], timeColumn: string, start: number | null, end: number | null, toTs: (value: unknown) => number | null): ParquetRow[] {
  if (!timeColumn || (start === null && end === null)) return rows;
  return rows.filter((row) => {
    const timestamp = toTs(row[timeColumn]);
    if (timestamp === null) return false;
    if (start !== null && timestamp < start) return false;
    if (end !== null && timestamp > end) return false;
    return true;
  });
}
