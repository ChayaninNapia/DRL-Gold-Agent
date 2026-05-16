import { useEffect, useMemo, useState } from 'react';
import { clampPage, filterRows } from '../lib/dataUtils';
import { valueToDisplay } from '../lib/formatting';
import type { ParquetRow } from '../types/parquet';

interface DataTableProps {
  rows: ParquetRow[];
  visibleColumns: string[];
}

const pageSizes = [25, 50, 100, 250];

export default function DataTable({ rows, visibleColumns }: DataTableProps) {
  const [searchText, setSearchText] = useState('');
  const [pageSize, setPageSize] = useState(50);
  const [page, setPage] = useState(1);

  const filteredRows = useMemo(() => filterRows(rows, searchText, visibleColumns), [rows, searchText, visibleColumns]);
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const currentPage = clampPage(page, totalPages);
  const pageRows = filteredRows.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  useEffect(() => {
    if (currentPage !== page) setPage(currentPage);
  }, [currentPage, page]);

  return (
    <section className="panel overflow-hidden">
      <div className="flex flex-col gap-3 border-b border-slate-700 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="font-semibold text-slate-50">Table Preview</h2>
          <p className="text-xs text-slate-400">{filteredRows.length.toLocaleString()} matching rows</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input className="control w-56" placeholder="Search visible columns..." value={searchText} onChange={(event) => {
            setSearchText(event.target.value);
            setPage(1);
          }} />
          <select className="control" value={pageSize} onChange={(event) => {
            setPageSize(Number(event.target.value));
            setPage(1);
          }}>
            {pageSizes.map((size) => (
              <option key={size} value={size}>
                {size} rows
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="max-h-[520px] overflow-auto">
        <table className="min-w-full border-separate border-spacing-0 text-left text-sm">
          <thead className="sticky top-0 z-10 bg-slate-800 text-xs uppercase text-slate-300">
            <tr>
              {visibleColumns.map((column) => (
                <th className="border-b border-slate-700 px-3 py-2 font-semibold" key={column}>
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 ? (
              <tr>
                <td className="px-3 py-8 text-center text-slate-400" colSpan={Math.max(1, visibleColumns.length)}>
                  No rows to display.
                </td>
              </tr>
            ) : (
              pageRows.map((row, rowIndex) => (
                <tr className="odd:bg-slate-900 even:bg-slate-950" key={`${currentPage}-${rowIndex}`}>
                  {visibleColumns.map((column) => (
                    <td className="max-w-[260px] whitespace-nowrap border-b border-slate-800 px-3 py-2 text-slate-200" key={column} title={valueToDisplay(row[column])}>
                      {valueToDisplay(row[column])}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className="flex flex-col gap-3 border-t border-slate-700 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="text-sm text-slate-300">
          Page {currentPage.toLocaleString()} / {totalPages.toLocaleString()}
        </div>
        <div className="flex gap-2">
          <button className="btn" type="button" disabled={currentPage <= 1} onClick={() => setPage(1)}>
            First
          </button>
          <button className="btn" type="button" disabled={currentPage <= 1} onClick={() => setPage(currentPage - 1)}>
            Prev
          </button>
          <button className="btn" type="button" disabled={currentPage >= totalPages} onClick={() => setPage(currentPage + 1)}>
            Next
          </button>
          <button className="btn" type="button" disabled={currentPage >= totalPages} onClick={() => setPage(totalPages)}>
            Last
          </button>
        </div>
      </div>
    </section>
  );
}
