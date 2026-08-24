import pandas as pd
import json
from pathlib import Path
from typing import Optional


def load_training_curves(log_path: str = "outputs/grpo-run-01/trainer_state.json") -> Optional[pd.DataFrame]:
    p = Path(log_path)
    if not p.exists():
        return None
    with open(p) as f:
        state = json.load(f)
    return pd.DataFrame(state.get("log_history", []))


def load_eval_results(path: str = "reports/eval_results.json") -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_sample_completions(path: str = "reports/samples.json") -> Optional[list]:
    """Expected format: [{"prompt": str, "before": str, "after": str}, ...]"""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)