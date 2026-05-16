import type { ColumnInfo, ParquetRow } from '../types/parquet';
import { toNumeric, toTimestamp, valueToDisplay } from './formatting';

const SAMPLE_LIMIT = 200;

export function normalizeRows(rawRows: Record<string, unknown>[]): ParquetRow[] {
  return rawRows.map((row) => {
    const normalized: ParquetRow = {};
    for (const [key, value] of Object.entries(row)) {
      if (typeof value === 'bigint') {
        const numberValue = Number(value);
        normalized[key] = Number.isSafeInteger(numberValue) ? numberValue : value.toString();
      } else {
        normalized[key] = value;
      }
    }
    return normalized;
  });
}

export function inferColumns(rows: ParquetRow[]): string[] {
  const columns: string[] = [];
  const seen = new Set<string>();

  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        seen.add(key);
        columns.push(key);
      }
    }
  }

  return columns;
}

export function inferSchema(rows: ParquetRow[], columns: string[]): ColumnInfo[] {
  return columns.map((name) => {
    let sample: unknown = undefined;
    let numericHits = 0;
    let timeHits = 0;
    let textHits = 0;
    let checked = 0;

    for (const row of rows) {
      const value = row[name];
      if (value === null || value === undefined || value === '') continue;
      if (sample === undefined) sample = value;

      checked += 1;
      if (toTimestamp(value) !== null) timeHits += 1;
      if (toNumeric(value) !== null) numericHits += 1;
      if (typeof value === 'string') textHits += 1;
      if (checked >= SAMPLE_LIMIT) break;
    }

    const lower = name.toLowerCase();
    let kind: ColumnInfo['kind'] = 'unknown';
    if (lower.includes('time') || (checked > 0 && timeHits / checked > 0.8 && numericHits / checked < 0.5)) {
      kind = 'time';
    } else if (checked > 0 && numericHits / checked > 0.8) {
      kind = 'numeric';
    } else if (checked > 0 && textHits > 0) {
      kind = 'text';
    }

    return { name, kind, sample };
  });
}

export function filterRows(rows: ParquetRow[], searchText: string, activeColumns: string[]): ParquetRow[] {
  const query = searchText.trim().toLowerCase();
  if (!query) return rows;

  return rows.filter((row) => {
    for (const column of activeColumns) {
      const display = valueToDisplay(row[column]).toLowerCase();
      if (display.includes(query)) return true;
    }
    return false;
  });
}

export function getNumericColumns(rows: ParquetRow[], columns: string[]): string[] {
  return columns.filter((column) => {
    let checked = 0;
    let numericHits = 0;
    for (const row of rows) {
      const value = row[column];
      if (value === null || value === undefined || value === '') continue;
      checked += 1;
      if (toNumeric(value) !== null) numericHits += 1;
      if (checked >= SAMPLE_LIMIT) break;
    }
    return checked > 0 && numericHits / checked > 0.8;
  });
}

export function findDefaultTimeColumn(columns: string[]): string {
  return columns.find((column) => column.toLowerCase() === 'time') ?? columns.find((column) => column.toLowerCase().includes('time')) ?? columns[0] ?? '';
}

export function clampPage(page: number, maxPage: number): number {
  if (!Number.isFinite(page)) return 1;
  if (maxPage < 1) return 1;
  if (page < 1) return 1;
  if (page > maxPage) return maxPage;
  return Math.floor(page);
}

export function missingExpectedColumns(columns: string[]): string[] {
  const expected = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'];
  const available = new Set(columns.map((column) => column.toLowerCase()));
  return expected.filter((column) => !available.has(column));
}
