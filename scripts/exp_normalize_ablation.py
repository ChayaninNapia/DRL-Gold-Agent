"""Ablation: does rolling z-score help, and does window size matter?

Runs three configs back-to-back, all else equal:
  1) exp_no_norm        — rolling_zscore_window = 0  (no normalization)
  2) exp_zscore_w60     — rolling_zscore_window = 60
  3) exp_zscore_w240    — rolling_zscore_window = 240

All other knobs come from config.yaml as-is. Writes a comparison summary at the end.
"""
import sys
import json
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import yaml

from train import train_ddqn


def run_one(base_cfg: dict, run_name: str, zscore_window: int) -> dict:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy via json
    cfg["features"]["rolling_zscore_window"] = zscore_window
    cfg["run"]["run_name"] = run_name
    print(f"\n{'=' * 70}\n[EXPERIMENT] {run_name}  (zscore_window={zscore_window})\n{'=' * 70}")
    t0 = time.time()
    test_m = train_ddqn(cfg)
    elapsed = time.time() - t0
    return {"run_name": run_name, "zscore_window": zscore_window, "elapsed_s": elapsed, "test": test_m}


def main():
    base_cfg = yaml.safe_load((WORKSPACE_ROOT / "config.yaml").read_text(encoding="utf-8"))

    experiments = [
        ("exp_no_norm", 0),
        ("exp_zscore_w60", 60),
        ("exp_zscore_w240", 240),
    ]

    results = []
    for name, win in experiments:
        results.append(run_one(base_cfg, name, win))

    # write comparison table
    out_dir = WORKSPACE_ROOT / "runs"
    summary_path = out_dir / "exp_normalize_ablation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n\n{'=' * 70}\n[ABLATION SUMMARY]  saved -> {summary_path}\n{'=' * 70}")
    print(f"{'run_name':<22} {'window':>8} {'test_ret':>10} {'sharpe':>9} {'sortino':>10} "
          f"{'mdd':>10} {'trades':>8} {'winrate':>9}")
    for r in results:
        m = r["test"]
        print(f"{r['run_name']:<22} {r['zscore_window']:>8} "
              f"{m['total_return']:>+10.4f} {m['sharpe']:>+9.3f} {m['sortino']:>+10.3f} "
              f"{m['mdd']:>+10.4f} {m['trades']:>8} {m['winrate']:>9.3f}")


if __name__ == "__main__":
    main()
