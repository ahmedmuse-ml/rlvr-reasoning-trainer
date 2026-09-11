import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.data.loader import load_multi_domain
from src.data.sql_loader import load_sql_dataset
from src.trainer.grpo_engine import build_grpo_trainer
from src.utils.logging_utils import get_logger

logger = get_logger("TrainPipeline")


def main():
    parser = argparse.ArgumentParser(
        description="RLVR Reasoning Trainer (GRPO / Dr. GRPO)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_config.yaml",
        help="Path to config YAML",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute 1-step CPU dry run to verify pipeline integrity",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Total number of training prompt samples",
    )

    parser.add_argument(
        "--sql",
        action="store_true",
        help="Train using the SQL application dataset",
    )

    args = parser.parse_args()

    # Select dataset.
    if args.sql:
        n_samples = min(
            4 if args.dry_run else args.samples,
            5,
        )

        logger.info(
            f"Loading SQL dataset ({n_samples} samples, "
            f"dry_run={args.dry_run})..."
        )

        dataset = load_sql_dataset(
            max_samples=n_samples,
        )

    else:
        n_samples = 4 if args.dry_run else args.samples

        logger.info(
            f"Loading dataset ({n_samples} samples, "
            f"dry_run={args.dry_run})..."
        )

        dataset = load_multi_domain(
            math_ratio=0.7,
            n_total=n_samples,
        )

    logger.info("Initializing Dr. GRPO Trainer...")

    trainer = build_grpo_trainer(
        args.config,
        dataset,
        dry_run=args.dry_run,
    )

    logger.info("Starting GRPO optimization loop...")

    trainer.train()

    # Save to a separate directory for SQL training.
    if args.sql:
        output_save_dir = (
            "outputs/sql_dry_run_model"
            if args.dry_run
            else "outputs/sql-grpo-final-model"
        )
    else:
        output_save_dir = (
            "outputs/dry_run_model"
            if args.dry_run
            else "outputs/grpo-final-model"
        )

    trainer.save_model(output_save_dir)

    logger.info(
        f"Training completed successfully! "
        f"Model weights saved to: {output_save_dir}"
    )


if __name__ == "__main__":
    main()