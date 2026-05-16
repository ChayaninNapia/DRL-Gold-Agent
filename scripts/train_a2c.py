"""A2C training per config.yaml."""
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import yaml

from train_a2c import train_a2c

cfg = yaml.safe_load((WORKSPACE_ROOT / "config.yaml").read_text(encoding="utf-8"))
# A2C runs are tracked separately from DDQN runs — adjust run_name if not overridden.
if cfg["run"]["run_name"].startswith("ddqn"):
    cfg["run"]["run_name"] = "a2c_v1"
train_a2c(cfg)
