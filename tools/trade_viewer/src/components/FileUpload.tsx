import { useRef } from 'react';

interface FileUploadProps {
  loading: boolean;
  error: string | null;
  onFileSelected: (file: File) => void;
}

export default function FileUpload({ loading, error, onFileSelected }: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  return (
    <section className="panel p-5">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-50">View Parquet</h1>
          <p className="mt-1 text-sm text-slate-400">View GOLD M1 parquet exports locally in your browser.</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <input
            ref={inputRef}
            className="hidden"
            type="file"
            accept=".parquet"
            onChange={(event) => {
              const file = event.currentTarget.files?.[0];
              if (file) onFileSelected(file);
            }}
          />
          <button className="btn btn-primary" type="button" disabled={loading} onClick={() => inputRef.current?.click()}>
            {loading ? 'Loading...' : 'Upload .parquet'}
          </button>
        </div>
      </div>
      {error ? <div className="mt-4 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">{error}</div> : null}
    </section>
  );
}
