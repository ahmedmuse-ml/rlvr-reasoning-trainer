import argparse
from src.data.loader import load_multi_domain
from src.trainer.grpo_engine import build_grpo_trainer
from src.utils.logging_utils import get_logger

logger = get_logger("TrainPipeline")


def main():
    parser = argparse.ArgumentParser(description="RLVR Reasoning Trainer (GRPO)")
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Fuli 1-step CPU dry-run si aad pipeline-ka u hubiso")
    parser.add_argument("--samples", type=int, default=1000, help="Tirada prompts-ka tababarka")
    args = parser.parse_args()

    n_samples = 4 if args.dry_run else args.samples
    logger.info(f"Soo dejinta dataset-ka ({n_samples} samples, dry_run={args.dry_run})...")
    dataset = load_multi_domain(math_ratio=0.7, n_total=n_samples)

    logger.info("Dhisidda GRPOTrainer...")
    trainer = build_grpo_trainer(args.config, dataset, dry_run=args.dry_run)

    logger.info("Bilowga tababarka GRPO...")
    trainer.train()

    output_save_dir = "outputs/dry_run_model" if args.dry_run else "outputs/grpo-final-model"
    trainer.save_model(output_save_dir)
    logger.info(f"Tababarku wuu dhamaaday! Model weights waxaa lagu keydiyay: {output_save_dir}")


if __name__ == "__main__":
    main()