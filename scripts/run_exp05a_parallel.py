"""Experiment 0.5a — Gamma sweep on DDQN (PROPOSAL.md Sec. 8).

Tests whether lowering gamma fixes the do-nothing collapse observed in Exp 0
(Zhang/Zohren/Roberts 2019 used gamma=0.3 for intraday DRL; ours used 0.99).

Sweep:  gamma in {0.3, 0.5, 0.9, 0.99} x seeds in {42, 1337, 2026} = 12 runs.
Reward: R4 (Exp 0 winner). Algorithm: DDQN only (Phase 1a; A2C/PPO Phase 1b).
Steps:  300k per run (~10 min/run sequential, ~5 min/run parallel N=4).

Parallelism:
  Launches up to MAX_PARALLEL subprocesses concurrently. Each subprocess
  runs train_ddqn() in isolation with PYTHONUNBUFFERED=1 + capped OMP/MKL
  threads to avoid CPU oversubscription. Smoke test showed N=4 is sweet
  spot (speedup 2.27x at 50k steps; expected ~2.5x at 300k).

Outputs (under runs/exp05a/):
  - <gamma>_s<seed>/                  one folder per run (best.pt, metrics.csv, ...)
  - exp05a_summary.csv                one row per run, sorted by gamma then seed
  - exp05a_collapse_report.json       collapse rate per gamma (val_trades < threshold)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

# Defaults — overridable via CLI
GAMMAS = [0.3, 0.5, 0.9, 0.99]
SEEDS = [42, 1337, 2026]
REWARD_MODE = "r4"
TOTAL_TIMESTEPS = 300_000
MAX_PARALLEL = 4
THREADS_PER_WORKER = 4  # 4 workers * 4 threads = 16, fits in 20 cores
RUN_PREFIX = "exp05a"
COLLAPSE_THRESHOLD = 50  # val_trades < this => degenerate "do-nothing" policy

METRICS = [
    "total_return", "mdd", "trades", "winrate",
    "sharpe", "sortino", "avg_trade_pnl", "turnover", "avg_holding_time",
    "final_equity", "max_dd_dollar", "max_dd_pct", "ruin_rate",
]


def _make_run_cfg(base_cfg: dict, gamma: float, seed: int, run_name: str,
                  total_timesteps: int, output_root: str) -> dict:
    """Build the per-run config (deep copy; no mutation of base_cfg)."""
    cfg = copy.deepcopy(base_cfg)
    cfg["train"]["seed"] = seed
    cfg["train"]["total_timesteps"] = int(total_timesteps)
    cfg["run"]["run_name"] = run_name
    cfg["run"]["output_root"] = output_root
    cfg["env"]["reward"]["mode"] = REWARD_MODE
    cfg["dqn"]["gamma"] = float(gamma)
    # Apply per-reward lr override for R4 (same as run_exp0.py does).
    lr_map = cfg["dqn"].get("lr_per_reward", {})
    if REWARD_MODE in lr_map:
        cfg["dqn"]["lr"] = float(lr_map[REWARD_MODE])
    # Strip _hpo (in case it slipped in from anywhere)
    cfg.pop("_hpo", None)
    return cfg


def _launch_worker(cfg: dict, run_name: str, log_path: Path) -> subprocess.Popen:
    """Launch one DDQN training subprocess with a temp config file."""
    python_exe = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"

    # Write the per-run config to a temp file so the child process can load it.
    tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".yaml",
                                      delete=False, prefix=f"{run_name}_")
    yaml.safe_dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.flush()
    tmp.close()
    cfg_path = tmp.name

    # Build inline python script: load config from temp file, call train_ddqn.
    script = (
        f"import sys, os, yaml\n"
        f"sys.path.insert(0, r'{WORKSPACE_ROOT / 'src'}')\n"
        f"os.environ['PYTHONUNBUFFERED'] = '1'\n"
        f"from pathlib import Path\n"
        f"cfg = yaml.safe_load(Path(r'{cfg_path}').read_text(encoding='utf-8'))\n"
        f"from train import train_ddqn\n"
        f"train_ddqn(cfg)\n"
        f"os.unlink(r'{cfg_path}')\n"
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
    env["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
    env["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_WORKER)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    p = subprocess.Popen(
        [str(python_exe), "-c", script],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )
    # Attach metadata for later inspection
    p._cfg_path = cfg_path  # type: ignore[attr-defined]
    p._log_fh = log_fh  # type: ignore[attr-defined]
    p._log_path = log_path  # type: ignore[attr-defined]
    return p


def _drain_completed(active: list[tuple], collected: list[dict],
                     out_root: Path, ranking_metric: str) -> None:
    """Poll active workers; for any that finished, collect results into `collected`.
    `active` is a list of (Popen, run_name, gamma, seed, t_start)."""
    still_active = []
    for entry in active:
        p, run_name, gamma, seed, t_start = entry
        if p.poll() is None:
            still_active.append(entry)
            continue
        # Worker finished
        p._log_fh.close()  # type: ignore[attr-defined]
        elapsed = time.time() - t_start
        rc = p.returncode
        if rc != 0:
            print(f"  [FAIL] {run_name} rc={rc} ({elapsed:.0f}s) — see {p._log_path}")
            collected.append({
                "gamma": gamma, "seed": seed, "run_name": run_name,
                "wall_sec": elapsed, "status": "FAIL", "return_code": rc,
                f"val_{ranking_metric}": float("nan"), "val_trades": 0,
                **{f"test_{m}": float("nan") for m in METRICS},
            })
            continue

        run_dir = out_root / run_name
        summary_path = run_dir / "summary.json"
        bi_path = run_dir / "best_info.json"
        val_metric_val = float("nan")
        val_trades = 0
        test_m = {}
        if bi_path.exists():
            try:
                bi = json.loads(bi_path.read_text(encoding="utf-8"))
                metric_name = bi.get("metric", ranking_metric)
                val_metric_val = float(bi.get(f"val_{metric_name}", float("nan")))
                val_trades = int(bi.get("val_trades", 0))
            except Exception as e:
                print(f"  [WARN] {run_name}: failed to parse best_info.json: {e}")
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                test_m = summary.get("test_metrics", {})
            except Exception as e:
                print(f"  [WARN] {run_name}: failed to parse summary.json: {e}")

        print(f"  [DONE] {run_name}  val_{ranking_metric}={val_metric_val:+.4f}  "
              f"val_trades={val_trades}  test_ret={test_m.get('total_return', float('nan')):+.4f}  "
              f"({elapsed:.0f}s)")

        row = {
            "gamma": gamma, "seed": seed, "run_name": run_name,
            "wall_sec": elapsed, "status": "OK", "return_code": 0,
            f"val_{ranking_metric}": val_metric_val,
            "val_trades": val_trades,
        }
        for m in METRICS:
            row[f"test_{m}"] = float(test_m.get(m, float("nan")))
        collected.append(row)

    active[:] = still_active


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 0.5a — gamma sweep on DDQN (parallel)")
    ap.add_argument("--gammas", type=float, nargs="+", default=GAMMAS,
                    help=f"Gamma values to sweep (default: {GAMMAS})")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                    help=f"Seeds (default: {SEEDS})")
    ap.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS,
                    help=f"Total timesteps per run (default: {TOTAL_TIMESTEPS})")
    ap.add_argument("--max-parallel", type=int, default=MAX_PARALLEL,
                    help=f"Max concurrent workers (default: {MAX_PARALLEL})")
    ap.add_argument("--run-prefix", default=RUN_PREFIX,
                    help=f"Output subfolder under runs/ (default: {RUN_PREFIX})")
    ap.add_argument("--config", default=str(WORKSPACE_ROOT / "config.yaml"),
                    help="Base config YAML (default: config.yaml)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip runs whose summary.json already exists (resume)")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_root = str(Path(base_cfg["run"]["output_root"]) / args.run_prefix).replace("\\", "/")
    out_root = (WORKSPACE_ROOT / output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ranking_metric = str(base_cfg["train"].get("best_metric", "total_return"))

    # Build the full work queue (gamma x seed)
    queue: list[tuple[float, int, str]] = []
    for gamma in args.gammas:
        for seed in args.seeds:
            run_name = f"g{gamma}_s{seed}"
            queue.append((gamma, seed, run_name))

    total_runs = len(queue)
    print(f"Experiment 0.5a — Gamma sweep (DDQN, R4)")
    print(f"  Gammas:         {args.gammas}")
    print(f"  Seeds:          {args.seeds}")
    print(f"  Total runs:     {total_runs}")
    print(f"  Steps/run:      {args.total_timesteps}")
    print(f"  Max parallel:   {args.max_parallel}")
    print(f"  Threads/worker: {THREADS_PER_WORKER}")
    print(f"  Output root:    {out_root}")
    print(f"  Per-run logs:   {log_dir}")
    print()

    collected: list[dict] = []
    active: list[tuple] = []  # (Popen, run_name, gamma, seed, t_start)
    t_global = time.time()
    queue_idx = 0

    while queue_idx < len(queue) or active:
        # Fill up to max_parallel
        while len(active) < args.max_parallel and queue_idx < len(queue):
            gamma, seed, run_name = queue[queue_idx]
            queue_idx += 1
            run_dir = out_root / run_name
            summary_path = run_dir / "summary.json"
            if args.skip_existing and summary_path.exists():
                print(f"  [SKIP] {run_name} (summary.json exists)")
                # Re-collect metrics from existing artifacts
                bi_path = run_dir / "best_info.json"
                val_metric_val = float("nan")
                val_trades = 0
                if bi_path.exists():
                    bi = json.loads(bi_path.read_text(encoding="utf-8"))
                    val_metric_val = float(bi.get(f"val_{ranking_metric}", float("nan")))
                    val_trades = int(bi.get("val_trades", 0))
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                test_m = summary.get("test_metrics", {})
                row = {
                    "gamma": gamma, "seed": seed, "run_name": run_name,
                    "wall_sec": 0.0, "status": "SKIP", "return_code": 0,
                    f"val_{ranking_metric}": val_metric_val, "val_trades": val_trades,
                }
                for m in METRICS:
                    row[f"test_{m}"] = float(test_m.get(m, float("nan")))
                collected.append(row)
                continue

            cfg = _make_run_cfg(base_cfg, gamma, seed, run_name,
                                args.total_timesteps, output_root)
            log_path = log_dir / f"{run_name}.log"
            t_start = time.time()
            p = _launch_worker(cfg, run_name, log_path)
            active.append((p, run_name, gamma, seed, t_start))
            print(f"  [LAUNCH {queue_idx}/{total_runs}] {run_name} (active={len(active)})")

        # Wait a bit, then drain completed
        if active:
            time.sleep(2)
            _drain_completed(active, collected, out_root, ranking_metric)

    total_elapsed = time.time() - t_global
    print(f"\nAll {total_runs} runs done in {total_elapsed/60:.1f} min")

    # Sort & write summary CSV
    collected.sort(key=lambda r: (r["gamma"], r["seed"]))
    summary_csv = out_root / "exp05a_summary.csv"
    fieldnames = (
        ["gamma", "seed", "run_name", "wall_sec", "status", "return_code",
         f"val_{ranking_metric}", "val_trades"]
        + [f"test_{m}" for m in METRICS]
    )
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(collected)
    print(f"Summary CSV     -> {summary_csv}")

    # Collapse audit: rate of (val_trades < threshold) per gamma
    print(f"\n{'-'*60}")
    print(f"Collapse audit (val_trades < {COLLAPSE_THRESHOLD} = collapse)")
    print(f"{'-'*60}")
    collapse_report = {}
    for gamma in args.gammas:
        rows_g = [r for r in collected if abs(r["gamma"] - gamma) < 1e-9]
        n = len(rows_g)
        n_collapse = sum(1 for r in rows_g if r["val_trades"] < COLLAPSE_THRESHOLD)
        ok_trades = [r["val_trades"] for r in rows_g if r["val_trades"] >= COLLAPSE_THRESHOLD]
        val_metrics = [r[f"val_{ranking_metric}"] for r in rows_g
                       if not (r[f"val_{ranking_metric}"] != r[f"val_{ranking_metric}"])]  # filter NaN
        mean_val = sum(val_metrics) / len(val_metrics) if val_metrics else float("nan")
        print(f"  gamma={gamma:<5} | n={n} | collapse={n_collapse}/{n} | "
              f"mean val_{ranking_metric}={mean_val:+.4f} | "
              f"mean trades (non-collapse)={sum(ok_trades)/max(1,len(ok_trades)):.0f}")
        collapse_report[str(gamma)] = {
            "n_runs": n, "n_collapse": n_collapse,
            "collapse_rate": n_collapse / n if n > 0 else float("nan"),
            f"mean_val_{ranking_metric}": mean_val,
            "mean_trades_non_collapse": sum(ok_trades) / max(1, len(ok_trades)),
        }
    report_path = out_root / "exp05a_collapse_report.json"
    report_path.write_text(json.dumps({
        "ranking_metric": ranking_metric,
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "by_gamma": collapse_report,
        "total_minutes": total_elapsed / 60.0,
    }, indent=2), encoding="utf-8")
    print(f"\nCollapse report -> {report_path}")


if __name__ == "__main__":
    main()
