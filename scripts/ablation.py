"""Run group-size / KL-coefficient / LoRA-vs-full-finetune ablations, log to W&B."""
import itertools
import yaml

ABLATION_GRID = {
    "num_generations": [4, 8, 16],
    "beta": [0.01, 0.04, 0.1],
}

def run_grid():
    keys, values = zip(*ABLATION_GRID.items())
    for combo in itertools.product(*values):
        overrides = dict(zip(keys, combo))
        print(f"Running ablation: {overrides}")
        # TODO: patch grpo_config.yaml with overrides, launch scripts/train.py subprocess

if __name__ == "__main__":
    run_grid()