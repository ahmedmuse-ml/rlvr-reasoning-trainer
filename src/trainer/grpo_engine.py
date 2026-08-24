import yaml
import torch
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from src.rewards.accuracy_reward import compute_accuracy_reward
from src.rewards.format_reward import compute_format_reward
from src.rewards.length_reward import compute_length_reward
from src.trainer.callbacks import RewardHackingMonitor


def get_lora_config(lora_cfg_path: str = "configs/lora_config.yaml") -> LoraConfig:
    with open(lora_cfg_path) as f:
        cfg = yaml.safe_load(f)

    return LoraConfig(
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def build_grpo_trainer(
    config_path: str,
    train_dataset: Dataset,
    dry_run: bool = False,
) -> GRPOTrainer:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 1. Hardware Detection (CPU vs GPU)
    is_cuda = torch.cuda.is_available()

    # 2. Model Selection (135M for lightweight CPU test, 1.5B for GPU)
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct" if dry_run else cfg["model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Precision Setting
    torch_dtype = torch.bfloat16 if is_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    # 4. Training Arguments
    num_generations = 2 if dry_run else cfg.get("num_generations", 8)
    per_device_train_batch_size = 2 if dry_run else cfg.get("per_device_train_batch_size", 2)
    gradient_accumulation_steps = 1 if dry_run else cfg.get("gradient_accumulation_steps", 4)
    max_steps = 1 if dry_run else -1
    use_vllm = False if (dry_run or not is_cuda) else cfg.get("use_vllm", True)


    training_args = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_generations=num_generations,
        max_completion_length=64 if dry_run else cfg.get("max_completion_length", 512),
        temperature=cfg.get("temperature", 0.8),
        beta=cfg.get("beta", 0.04),
        learning_rate=float(cfg.get("learning_rate", 1e-6)),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=50,
        report_to="none" if dry_run else cfg.get("report_to", "none"),
        use_vllm=use_vllm,
        # Explicit CPU/GPU isolation flags:
        use_cpu=not is_cuda,
        bf16=is_cuda,
        fp16=False,
    )

    lora_config = get_lora_config()

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        reward_funcs=[
            compute_accuracy_reward,
            compute_format_reward,
            compute_length_reward,
        ],
        callbacks=[RewardHackingMonitor()],
    )

    return trainer
