import { describe, expect, it } from 'vitest';
import { defaultLineY, defaultScatterX, defaultScatterY, downsampleRows, getExtent } from './chartUtils';
import { clampPage, findDefaultTimeColumn } from './dataUtils';
import { toNumeric, toTimestamp, valueToDisplay } from './formatting';

describe('formatting utilities', () => {
  it('displays non-integer decimals with two places', () => {
    expect(valueToDisplay(4617.2)).toBe('4617.20');
    expect(valueToDisplay(2642.69)).toBe('2642.69');
  });

  it('displays integers without decimals', () => {
    expect(valueToDisplay(264)).toBe('264');
  });

  it('accepts numeric strings and rejects text', () => {
    expect(toNumeric('4617.20')).toBe(4617.2);
    expect(toNumeric('text')).toBeNull();
  });

  it('parses ISO datetime values', () => {
    expect(toTimestamp('2026-05-01T00:00:00.000Z')).toBe(Date.parse('2026-05-01T00:00:00.000Z'));
  });
});

describe('data utilities', () => {
  it('clamps page underflow and overflow', () => {
    expect(clampPage(-4, 10)).toBe(1);
    expect(clampPage(20, 10)).toBe(10);
    expect(clampPage(4, 10)).toBe(4);
  });

  it('selects default fields by MT5 preference', () => {
    const columns = ['time', 'open', 'close', 'tick_volume', 'spread'];
    const numeric = ['open', 'close', 'tick_volume', 'spread'];
    expect(findDefaultTimeColumn(columns)).toBe('time');
    expect(defaultLineY(columns, numeric)).toBe('close');
    expect(defaultScatterX(numeric)).toBe('tick_volume');
    expect(defaultScatterY(numeric)).toBe('spread');
  });
});

describe('chart utilities', () => {
  it('gets extent for 100,000 values without stack overflow', () => {
    const values = Array.from({ length: 100_000 }, (_, index) => index - 500);
    expect(getExtent(values)).toEqual([-500, 99_499]);
  });

  it('limits downsampled rows', () => {
    const rows = Array.from({ length: 20_000 }, (_, index) => ({ index }));
    const sampled = downsampleRows(rows, 5000);
    expect(sampled.length).toBeLessThanOrEqual(5000);
    expect(sampled[0]).toEqual({ index: 0 });
    expect(sampled[sampled.length - 1]).toEqual({ index: 19_999 });
  });
});
