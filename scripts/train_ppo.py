"""PPO training per config.yaml."""
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import yaml

from train_ppo import train_ppo

cfg = yaml.safe_load((WORKSPACE_ROOT / "config.yaml").read_text(encoding="utf-8"))
# PPO runs are tracked separately from DDQN/A2C runs — adjust run_name if not overridden.
if cfg["run"]["run_name"].startswith("ddqn") or cfg["run"]["run_name"].startswith("a2c"):
    cfg["run"]["run_name"] = "ppo_v1"
train_ppo(cfg)
