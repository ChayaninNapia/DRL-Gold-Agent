"""Quick sanity: tiny DDQN run to verify the pipeline end-to-end."""
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import yaml

from train import train_ddqn

cfg = yaml.safe_load((WORKSPACE_ROOT / "config.yaml").read_text(encoding="utf-8"))
cfg["train"]["total_timesteps"] = 20000
cfg["dqn"]["learning_starts"] = 500
cfg["dqn"]["buffer_size"] = 10000
cfg["train"]["early_stop_patience"] = 99
cfg["train"]["eval_every_sessions"] = 5  # tighter eval for short sanity run
cfg["run"]["run_name"] = "sanity"

train_ddqn(cfg)
