import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.data.loader import load_multi_domain
from src.trainer.grpo_engine import build_grpo_trainer
from src.utils.logging_utils import get_logger

logger = get_logger("TrainPipeline")


def main():
    parser = argparse.ArgumentParser(description="RLVR Reasoning Trainer (GRPO / Dr. GRPO)")
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Execute 1-step CPU dry run to verify pipeline integrity")
    parser.add_argument("--samples", type=int, default=1000, help="Total number of training prompt samples")
    args = parser.parse_args()

    n_samples = 4 if args.dry_run else args.samples
    logger.info(f"Loading dataset ({n_samples} samples, dry_run={args.dry_run})...")
    dataset = load_multi_domain(math_ratio=0.7, n_total=n_samples)

    logger.info("Initializing Dr. GRPO Trainer...")
    trainer = build_grpo_trainer(args.config, dataset, dry_run=args.dry_run)

    logger.info("Starting GRPO optimization loop...")
    trainer.train()

    output_save_dir = "outputs/dry_run_model" if args.dry_run else "outputs/grpo-final-model"
    trainer.save_model(output_save_dir)
    logger.info(f"Training completed successfully! Model weights saved to: {output_save_dir}")


if __name__ == "__main__":
    main()