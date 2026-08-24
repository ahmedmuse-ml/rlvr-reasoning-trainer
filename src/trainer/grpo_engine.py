import yaml
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rewards.accuracy_reward import compute_accuracy_reward
from src.rewards.format_reward import compute_format_reward
from src.rewards.length_reward import compute_length_reward
from src.trainer.callbacks import RewardHackingMonitor


def build_trainer(config_path: str, train_dataset):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])

    grpo_config = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_generations=cfg["num_generations"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        beta=cfg["beta"],
        epsilon=cfg["epsilon"],
        epsilon_high=cfg.get("epsilon_high", cfg["epsilon"]),
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        reward_weights=cfg["reward_weights"],
        use_vllm=cfg["use_vllm"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        report_to=cfg["report_to"],
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[compute_accuracy_reward, compute_format_reward, compute_length_reward],
        callbacks=[RewardHackingMonitor()],
    )
    return trainer