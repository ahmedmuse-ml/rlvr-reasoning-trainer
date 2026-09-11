import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from src.data.loader import load_math_dataset, load_code_dataset
from src.data.sql_loader import load_sql_dataset
from src.evaluation.evaluate import Evaluator
from src.utils.logging_utils import get_logger


logger = get_logger("Evaluation")


def generate_completion(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens: int = 256,
) -> str:
    """
    Generate one completion for a structured chat prompt.
    """
    prompt_text = tokenizer.apply_chat_template(
        prompt,
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
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


def run_evaluation(
    model_path: str,
    is_base: bool = False,
    n_samples: int = 2,
    sql: bool = False,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float16
    )

    logger.info(
        f"Loading evaluation model from: {model_path}..."
    )

    if is_base:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)

    else:
        # Automatically detect base model from adapter config.
        try:
            peft_config = PeftConfig.from_pretrained(
                model_path
            )
            base_model_id = (
                peft_config.base_model_name_or_path
            )
        except Exception:
            base_model_id = model_path

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id
        )

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

    # ---------------------------------------------------------
    # SQL evaluation mode
    # ---------------------------------------------------------
    if sql:
        logger.info(
            f"Evaluating SQL application dataset "
            f"({n_samples} samples)..."
        )

        sql_ds = load_sql_dataset(
            max_samples=n_samples,
        )

        evaluation_tasks = []

        for item in sql_ds:
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=item["prompt"],
                device=device,
                max_new_tokens=256,
            )

            evaluation_tasks.append(
                {
                    "completion": completion,
                    "domain": "sql",
                    "answer": "",
                    "test_list": [],
                    "database_sql": item["database_sql"],
                    "reference_sql": item["reference_sql"],
                }
            )

        results = evaluator.evaluate(
            model=model,
            tasks=evaluation_tasks,
        )

        metrics = {
            "model": model_path,
            "sql_accuracy": round(
                results["sql_accuracy"] * 100,
                2,
            ),
            "total": results["total"],
        }

        Path("reports").mkdir(exist_ok=True)

        report_file = (
            "reports/sql_baseline_results.json"
            if is_base
            else "reports/sql_eval_results.json"
        )

        with open(report_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            f"SQL evaluation complete! Results: {metrics}"
        )

        return metrics

    # ---------------------------------------------------------
    # Existing Math + Code evaluation mode
    # ---------------------------------------------------------
    evaluation_tasks = []

    # 1. Generate Math Benchmark (GSM8K) completions.
    logger.info(
        f"Evaluating GSM8K Math Benchmark "
        f"({n_samples} samples)..."
    )

    math_ds = load_math_dataset(
        split="test",
        max_samples=n_samples,
    )

    for item in math_ds:
        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=item["prompt"],
            device=device,
            max_new_tokens=256,
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
        f"Evaluating MBPP Code Benchmark "
        f"({n_samples} samples)..."
    )

    code_ds = load_code_dataset(
        split="test",
        max_samples=n_samples,
    )

    for item in code_ds:
        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=item["prompt"],
            device=device,
            max_new_tokens=256,
        )

        evaluation_tasks.append(
            {
                "completion": completion,
                "domain": "code",
                "answer": "",
                "test_list": item["test_list"],
            }
        )

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

    logger.info(
        f"Evaluation complete! Results: {metrics}"
    )

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

    parser.add_argument(
        "--sql",
        action="store_true",
        help="Evaluate the SQL application dataset",
    )

    args = parser.parse_args()

    run_evaluation(
        args.model,
        is_base=args.base,
        n_samples=args.samples,
        sql=args.sql,
    )