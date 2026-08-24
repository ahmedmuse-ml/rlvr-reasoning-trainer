from src.data.loader import load_multi_domain
from src.trainer.grpo_engine import build_trainer

def main():
    dataset = load_multi_domain(math_ratio=0.5, n_total=2000)
    trainer = build_trainer("configs/grpo_config.yaml", dataset)
    trainer.train()
    trainer.save_model("outputs/grpo-run-01/final")

if __name__ == "__main__":
    main()