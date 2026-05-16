import { useState } from 'react';
import ParquetView from './ParquetView';
import TradeHistoryView from './TradeHistoryView';

type Tab = 'parquet' | 'trades';

export default function App() {
  const [tab, setTab] = useState<Tab>('parquet');

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-5 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-[1680px] flex-col gap-4">
        <nav className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900 p-2">
          <button
            type="button"
            className={`tab ${tab === 'parquet' ? 'tab-active' : ''}`}
            onClick={() => setTab('parquet')}
          >
            View Parquet
          </button>
          <button
            type="button"
            className={`tab ${tab === 'trades' ? 'tab-active' : ''}`}
            onClick={() => setTab('trades')}
          >
            Trade History
          </button>
        </nav>

        {tab === 'parquet' ? <ParquetView /> : <TradeHistoryView />}
      </div>
    </main>
  );
}
