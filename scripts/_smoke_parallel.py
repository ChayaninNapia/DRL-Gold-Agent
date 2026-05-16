"""Smoke test: launch N parallel DDQN runs and measure wall time.

Goal: compare sequential vs parallel for Phase 1a screen (12 DDQN runs).
Runs short 5k-step training so total wall time stays under a few minutes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def make_worker_script(run_name: str, seed: int, gamma: float, total_steps: int) -> str:
    """Build the inline python -c script for one worker."""
    return f"""
import sys, os
sys.path.insert(0, r'{WORKSPACE_ROOT / "src"}')
os.environ['PYTHONUNBUFFERED'] = '1'
import yaml, copy
from pathlib import Path

cfg = yaml.safe_load(Path(r'{WORKSPACE_ROOT / "config.yaml"}').read_text(encoding='utf-8'))
cfg = copy.deepcopy(cfg)
cfg['train']['total_timesteps'] = {total_steps}
cfg['train']['seed'] = {seed}
cfg['run']['output_root'] = 'runs/_smoke_parallel'
cfg['run']['run_name'] = {run_name!r}
cfg['env']['reward']['mode'] = 'r4'
cfg['dqn']['lr'] = 1.0e-5
cfg['dqn']['gamma'] = {gamma}
cfg['train']['eval_every_sessions'] = 100
cfg['train']['early_stop_patience'] = 100

from train import train_ddqn
train_ddqn(cfg)
"""


def launch_parallel(n_workers: int, total_steps: int, threads_per_worker: int = 0) -> float:
    """Launch N workers, wait, return wall time.

    threads_per_worker: cap torch/OMP/MKL threads to this. 0 = no cap (default).
    Important on Windows: PyTorch defaults to num_cores threads per process,
    so N>1 processes oversubscribe CPU and lose almost all parallelism gain.
    """
    python_exe = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"
    procs = []
    t0 = time.time()
    for i in range(n_workers):
        gamma = 0.3 + i * 0.2  # vary gamma so runs aren't identical
        script = make_worker_script(
            run_name=f"par_w{i}_g{gamma:.1f}",
            seed=42 + i,
            gamma=gamma,
            total_steps=total_steps,
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if threads_per_worker > 0:
            env["OMP_NUM_THREADS"] = str(threads_per_worker)
            env["MKL_NUM_THREADS"] = str(threads_per_worker)
            env["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)
        # Pipe stdout to DEVNULL to avoid console interleaving
        p = subprocess.Popen(
            [str(python_exe), "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        procs.append(p)

    for p in procs:
        p.wait()

    elapsed = time.time() - t0

    # Surface any worker errors
    for i, p in enumerate(procs):
        if p.returncode != 0:
            err = p.stderr.read().decode("utf-8", errors="replace") if p.stderr else "(no stderr)"
            print(f"  Worker {i} FAILED rc={p.returncode}:\n{err[-500:]}")

    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="Number of parallel workers to test")
    ap.add_argument("--steps", type=int, default=5000,
                    help="Total timesteps per worker run")
    ap.add_argument("--threads", type=int, default=0,
                    help="Cap torch/OMP/MKL threads per worker (0=no cap). "
                         "Recommended: floor(n_cores / max_workers).")
    args = ap.parse_args()

    print(f"Smoke parallel test")
    print(f"  Steps per worker: {args.steps}")
    print(f"  Worker counts:    {args.workers}")
    print()

    print(f"  Threads/worker:   {args.threads if args.threads > 0 else 'no cap (default ~n_cores)'}")
    print()

    results = []
    for n in args.workers:
        print(f"=== Testing N={n} parallel workers ===")
        elapsed = launch_parallel(n, args.steps, threads_per_worker=args.threads)
        per_run = elapsed / n
        print(f"  Wall time: {elapsed:.1f}s  ({n} runs, {per_run:.1f}s per run)")
        results.append((n, elapsed, per_run))

    print(f"\n{'='*60}")
    print(f"Summary (lower per-run wall is better)")
    print(f"{'='*60}")
    print(f"{'N':>3} | {'wall (s)':>10} | {'per-run (s)':>12} | {'speedup vs N=1':>15}")
    base = results[0][2] if results else 1.0
    for n, wall, per in results:
        speedup = base / per if per > 0 else 0
        print(f"{n:>3} | {wall:>10.1f} | {per:>12.1f} | {speedup:>15.2f}x")


if __name__ == "__main__":
    main()
