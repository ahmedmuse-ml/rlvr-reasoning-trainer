import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from src.data.loader import load_math_dataset, load_code_dataset
from src.evaluation.evaluate import Evaluator
from src.utils.logging_utils import get_logger


logger = get_logger("Evaluation")


def run_evaluation(
    model_path: str,
    is_base: bool = False,
    n_samples: int = 2,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = (
        torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    logger.info(f"Loading evaluation model from: {model_path}...")

    if is_base:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        # Automatically detect base model from adapter config.
        try:
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_id = peft_config.base_model_name_or_path
        except Exception:
            base_model_id = model_path

        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)

        try:
            model = PeftModel.from_pretrained(
                base_model,
                model_path,
            ).to(device)
        except Exception:
            model = base_model

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    evaluator = Evaluator()
    evaluation_tasks = []

    # 1. Generate Math Benchmark (GSM8K) completions.
    logger.info(
        f"Evaluating GSM8K Math Benchmark ({n_samples} samples)..."
    )

    math_ds = load_math_dataset(
        split="test",
        max_samples=n_samples,
    )

    for item in math_ds:
        prompt_text = tokenizer.apply_chat_template(
            item["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        completion = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        evaluation_tasks.append(
            {
                "completion": completion,
                "domain": "math",
                "answer": item["answer"],
                "test_list": [],
            }
        )

    # 2. Generate Code Benchmark (MBPP) completions.
    logger.info(
        f"Evaluating MBPP Code Benchmark ({n_samples} samples)..."
    )

    code_ds = load_code_dataset(
        split="test",
        max_samples=n_samples,
    )

    for item in code_ds:
        prompt_text = tokenizer.apply_chat_template(
            item["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        completion = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        evaluation_tasks.append(
            {
                "completion": completion,
                "domain": "code",
                "answer": "",
                "test_list": item["test_list"],
            }
        )

    # Evaluate all generated completions through the framework evaluator.
    results = evaluator.evaluate(
        model=model,
        tasks=evaluation_tasks,
    )

    metrics = {
        "model": model_path,
        "gsm8k_accuracy": round(
            results["math_accuracy"] * 100,
            2,
        ),
        "mbpp_pass@1": round(
            results["code_accuracy"] * 100,
            2,
        ),
    }

    Path("reports").mkdir(exist_ok=True)

    report_file = "reports/eval_results.json"

    with open(report_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation complete! Results: {metrics}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="outputs/dry_run_model",
    )

    parser.add_argument(
        "--base",
        action="store_true",
        help="Evaluate base model instead of fine-tuned adapter",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    run_evaluation(
        args.model,
        is_base=args.base,
        n_samples=args.samples,
    )