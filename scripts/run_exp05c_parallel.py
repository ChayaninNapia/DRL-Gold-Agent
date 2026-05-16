"""Experiment 0.5c -- A2C/PPO collapse fix via 4-knob config tune (post-Phase-1b audit).

Phase 1b (Exp 0.5b) showed gamma=0.30 does NOT generalize to A2C/PPO -- they
collapse to uniform-random policies (entropy H=1.099=ln(3), KL=0 every update,
EV=0). Sub-agent code audit (3 independent agents converged on the same root
cause) identified 4 config-level bugs that interact with gamma=0.30:

  Bug 1: gae_lambda=0.95 -> at gamma=0.30, gamma*lambda=0.285 cuts off after
         ~3 bars. Advantages become per-bar noise. Fix: gae_lambda=1.0 (MC).
  Bug 2: value_coef=0.5 -> 99% of total loss is value MSE; policy head starves.
         Fix: value_coef=0.25.
  Bug 3: entropy_coef=0.01 -> too weak to maintain exploration at short horizon.
         Fix: entropy_coef=0.05.
  Bug 4: PPO n_epochs=10 -> with noisy advantages, 100+ SGD steps churn the
         value head while policy stays put. Fix: n_epochs=4.

Sweep: 2 algos {A2C, PPO} x 3 seeds {42, 1337, 2026} x gamma=0.3 x R4 = 6 runs.
Same 300k steps and parallel N=4 as Phase 1b -- minimum-change A/B vs 1b.

Outputs (under runs/exp05c/):
  - <algo>_s<seed>/                   per-run artifacts
  - exp05c_summary.csv                one row per run
  - exp05c_report.json                collapse rate + mean metrics per algo
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

ALGOS = ["a2c", "ppo"]
SEEDS = [42, 1337, 2026]
GAMMA = 0.30
REWARD_MODE = "r4"
TOTAL_TIMESTEPS = 300_000
MAX_PARALLEL = 4
THREADS_PER_WORKER = 4
RUN_PREFIX = "exp05c"
COLLAPSE_THRESHOLD = 50

# --- 4-knob config fix from Agent 3 audit (2026-05-16) ---
# These override the values in config.yaml for A2C and PPO in this phase only.
A2C_OVERRIDES = {
    "gae_lambda": 1.0,     # was 0.95 -- bug #1 (MC returns at gamma=0.3)
    "value_coef": 0.25,    # was 0.50 -- bug #2 (don't drown policy head)
    "entropy_coef": 0.05,  # was 0.01 -- bug #3 (more exploration at short horizon)
}
PPO_OVERRIDES = {
    "gae_lambda": 1.0,
    "value_coef": 0.25,
    "entropy_coef": 0.05,
    "n_epochs": 4,         # was 10   -- bug #4 (don't over-SGD on noisy advantages)
}

METRICS = [
    "total_return", "mdd", "trades", "winrate",
    "sharpe", "sortino", "avg_trade_pnl", "turnover", "avg_holding_time",
    "final_equity", "max_dd_dollar", "max_dd_pct", "ruin_rate",
]

_ALGO_TRAIN_FN = {
    "a2c": ("train_a2c", "train_a2c"),
    "ppo": ("train_ppo", "train_ppo"),
}
_ALGO_CFG_SECTION = {"a2c": "a2c", "ppo": "ppo"}
_ALGO_OVERRIDES = {"a2c": A2C_OVERRIDES, "ppo": PPO_OVERRIDES}


def _make_run_cfg(base_cfg: dict, algo: str, seed: int, run_name: str,
                  total_timesteps: int, output_root: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["train"]["seed"] = seed
    cfg["train"]["total_timesteps"] = int(total_timesteps)
    cfg["run"]["run_name"] = run_name
    cfg["run"]["output_root"] = output_root
    cfg["env"]["reward"]["mode"] = REWARD_MODE

    section = cfg[_ALGO_CFG_SECTION[algo]]
    section["gamma"] = float(GAMMA)

    # Apply 4-knob overrides (Phase 1c key change)
    for k, v in _ALGO_OVERRIDES[algo].items():
        section[k] = v

    # Apply per-reward lr (same as Phase 1b)
    lr_map = section.get("lr_per_reward", {})
    if REWARD_MODE in lr_map:
        section["lr"] = float(lr_map[REWARD_MODE])

    cfg.pop("_hpo", None)
    return cfg


def _launch_worker(cfg: dict, algo: str, run_name: str, log_path: Path) -> subprocess.Popen:
    python_exe = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"

    tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".yaml",
                                      delete=False, prefix=f"{run_name}_")
    yaml.safe_dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.flush()
    tmp.close()
    cfg_path = tmp.name

    module, fn = _ALGO_TRAIN_FN[algo]
    script = (
        f"import sys, os, yaml\n"
        f"sys.path.insert(0, r'{WORKSPACE_ROOT / 'src'}')\n"
        f"os.environ['PYTHONUNBUFFERED'] = '1'\n"
        f"from pathlib import Path\n"
        f"cfg = yaml.safe_load(Path(r'{cfg_path}').read_text(encoding='utf-8'))\n"
        f"from {module} import {fn}\n"
        f"{fn}(cfg)\n"
        f"os.unlink(r'{cfg_path}')\n"
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
    env["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
    env["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_WORKER)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    p = subprocess.Popen(
        [str(python_exe), "-c", script],
        stdout=log_fh, stderr=subprocess.STDOUT, env=env,
    )
    p._cfg_path = cfg_path  # type: ignore[attr-defined]
    p._log_fh = log_fh  # type: ignore[attr-defined]
    p._log_path = log_path  # type: ignore[attr-defined]
    return p


def _drain_completed(active: list[tuple], collected: list[dict],
                     out_root: Path, ranking_metric: str) -> None:
    still_active = []
    for entry in active:
        p, run_name, algo, seed, t_start = entry
        if p.poll() is None:
            still_active.append(entry)
            continue
        p._log_fh.close()  # type: ignore[attr-defined]
        elapsed = time.time() - t_start
        rc = p.returncode
        if rc != 0:
            print(f"  [FAIL] {run_name} rc={rc} ({elapsed:.0f}s) -- see {p._log_path}")
            collected.append({
                "algo": algo, "seed": seed, "run_name": run_name,
                "wall_sec": elapsed, "status": "FAIL", "return_code": rc,
                f"val_{ranking_metric}": float("nan"), "val_trades": 0,
                **{f"test_{m}": float("nan") for m in METRICS},
            })
            continue

        run_dir = out_root / run_name
        bi_path = run_dir / "best_info.json"
        summary_path = run_dir / "summary.json"
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
                print(f"  [WARN] {run_name}: best_info.json: {e}")
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                test_m = summary.get("test_metrics", {})
            except Exception as e:
                print(f"  [WARN] {run_name}: summary.json: {e}")

        print(f"  [DONE] {run_name}  val_{ranking_metric}={val_metric_val:+.4f}  "
              f"val_trades={val_trades}  test_ret={test_m.get('total_return', float('nan')):+.4f}  "
              f"({elapsed:.0f}s)")

        row = {
            "algo": algo, "seed": seed, "run_name": run_name,
            "wall_sec": elapsed, "status": "OK", "return_code": 0,
            f"val_{ranking_metric}": val_metric_val, "val_trades": val_trades,
        }
        for m in METRICS:
            row[f"test_{m}"] = float(test_m.get(m, float("nan")))
        collected.append(row)
    active[:] = still_active


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 0.5c -- A2C/PPO collapse fix")
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    ap.add_argument("--max-parallel", type=int, default=MAX_PARALLEL)
    ap.add_argument("--run-prefix", default=RUN_PREFIX)
    ap.add_argument("--config", default=str(WORKSPACE_ROOT / "config.yaml"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_root = str(Path(base_cfg["run"]["output_root"]) / args.run_prefix).replace("\\", "/")
    out_root = (WORKSPACE_ROOT / output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ranking_metric = str(base_cfg["train"].get("best_metric", "total_return"))

    queue: list[tuple[str, int, str]] = []
    for algo in args.algos:
        for seed in args.seeds:
            queue.append((algo, seed, f"{algo}_s{seed}"))

    total_runs = len(queue)
    print(f"Experiment 0.5c -- A2C/PPO 4-knob fix (gae_lambda=1.0, value_coef=0.25,")
    print(f"                                       entropy_coef=0.05, ppo.n_epochs=4)")
    print(f"  Algos:          {args.algos}")
    print(f"  Seeds:          {args.seeds}")
    print(f"  gamma:          {GAMMA}")
    print(f"  Reward:         {REWARD_MODE}")
    print(f"  A2C overrides:  {A2C_OVERRIDES}")
    print(f"  PPO overrides:  {PPO_OVERRIDES}")
    print(f"  Total runs:     {total_runs}")
    print(f"  Steps/run:      {args.total_timesteps}")
    print(f"  Max parallel:   {args.max_parallel}")
    print(f"  Threads/worker: {THREADS_PER_WORKER}")
    print(f"  Output root:    {out_root}")
    print()

    collected: list[dict] = []
    active: list[tuple] = []
    t_global = time.time()
    queue_idx = 0

    while queue_idx < len(queue) or active:
        while len(active) < args.max_parallel and queue_idx < len(queue):
            algo, seed, run_name = queue[queue_idx]
            queue_idx += 1
            run_dir = out_root / run_name
            summary_path = run_dir / "summary.json"
            if args.skip_existing and summary_path.exists():
                print(f"  [SKIP] {run_name}")
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
                    "algo": algo, "seed": seed, "run_name": run_name,
                    "wall_sec": 0.0, "status": "SKIP", "return_code": 0,
                    f"val_{ranking_metric}": val_metric_val, "val_trades": val_trades,
                }
                for m in METRICS:
                    row[f"test_{m}"] = float(test_m.get(m, float("nan")))
                collected.append(row)
                continue

            cfg = _make_run_cfg(base_cfg, algo, seed, run_name,
                                args.total_timesteps, output_root)
            log_path = log_dir / f"{run_name}.log"
            t_start = time.time()
            p = _launch_worker(cfg, algo, run_name, log_path)
            active.append((p, run_name, algo, seed, t_start))
            print(f"  [LAUNCH {queue_idx}/{total_runs}] {run_name} (active={len(active)})")

        if active:
            time.sleep(2)
            _drain_completed(active, collected, out_root, ranking_metric)

    total_elapsed = time.time() - t_global
    print(f"\nAll {total_runs} runs done in {total_elapsed/60:.1f} min")

    collected.sort(key=lambda r: (r["algo"], r["seed"]))
    summary_csv = out_root / "exp05c_summary.csv"
    fieldnames = (
        ["algo", "seed", "run_name", "wall_sec", "status", "return_code",
         f"val_{ranking_metric}", "val_trades"]
        + [f"test_{m}" for m in METRICS]
    )
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(collected)
    print(f"Summary CSV     -> {summary_csv}")

    print(f"\n{'-'*60}")
    print(f"Collapse audit (val_trades < {COLLAPSE_THRESHOLD} = collapse)")
    print(f"{'-'*60}")
    report = {}
    for algo in args.algos:
        rows_a = [r for r in collected if r["algo"] == algo]
        n = len(rows_a)
        n_collapse = sum(1 for r in rows_a if r["val_trades"] < COLLAPSE_THRESHOLD)
        val_metrics = [r[f"val_{ranking_metric}"] for r in rows_a
                       if not (r[f"val_{ranking_metric}"] != r[f"val_{ranking_metric}"])]
        test_rets = [r["test_total_return"] for r in rows_a
                     if not (r["test_total_return"] != r["test_total_return"])]
        sortinos = [r["test_sortino"] for r in rows_a
                    if not (r["test_sortino"] != r["test_sortino"])]
        mean_val = sum(val_metrics) / len(val_metrics) if val_metrics else float("nan")
        mean_test = sum(test_rets) / len(test_rets) if test_rets else float("nan")
        mean_sortino = sum(sortinos) / len(sortinos) if sortinos else float("nan")
        print(f"  algo={algo:<4} | n={n} | collapse={n_collapse}/{n} | "
              f"mean val_{ranking_metric}={mean_val:+.4f} | "
              f"mean test_ret={mean_test:+.4f} | mean Sortino={mean_sortino:+.2f}")
        report[algo] = {
            "n_runs": n, "n_collapse": n_collapse,
            "collapse_rate": n_collapse / n if n > 0 else float("nan"),
            f"mean_val_{ranking_metric}": mean_val,
            "mean_test_return": mean_test,
            "mean_test_sortino": mean_sortino,
        }

    report_path = out_root / "exp05c_report.json"
    report_path.write_text(json.dumps({
        "gamma": GAMMA, "reward_mode": REWARD_MODE,
        "a2c_overrides": A2C_OVERRIDES, "ppo_overrides": PPO_OVERRIDES,
        "ranking_metric": ranking_metric,
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "by_algo": report,
        "total_minutes": total_elapsed / 60.0,
    }, indent=2), encoding="utf-8")
    print(f"\nReport          -> {report_path}")


if __name__ == "__main__":
    main()
