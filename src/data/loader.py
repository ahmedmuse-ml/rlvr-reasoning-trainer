from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset, concatenate_datasets


def _format_math_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
    """GSM8K formatter oo ku xidhaya chat template iyo XML tags adag."""
    raw_answer = example["answer"]
    clean_answer = raw_answer.split("####")[-1].strip() if "####" in raw_answer else raw_answer.strip()

    return {
        "prompt": [
            {
                "role": "system",
                "content": (
                    "Solve the following mathematical problem step by step. "
                    "Structure your reasoning inside <think>...</think> tags and provide "
                    "the exact final answer inside <answer>...</answer> tags."
                ),
            },
            {"role": "user", "content": example["question"]},
        ],
        "answer": clean_answer,
        "test_list": [],
        "domain": "math",
    }


def _format_code_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
    """MBPP formatter oo ku xidhaya Python code prompt."""
    task_description = example.get("prompt") or example.get("text", "")

    return {
        "prompt": [
            {
                "role": "system",
                "content": (
                    "Write a Python function to solve the given task. "
                    "First explain your logic inside <think>...</think> tags, then provide "
                    "the executable Python code inside <answer>```python\n...\n```</answer> tags."
                ),
            },
            {"role": "user", "content": task_description},
        ],
        "answer": "",
        "test_list": example.get("test_list", []),
        "domain": "code",
    }


def load_math_dataset(split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    """Soo deji GSM8K adoo ilaalinaya columns-ka midaysan."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.map(_format_math_prompt, remove_columns=ds.column_names)
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds


def load_code_dataset(split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    """Soo deji MBPP sanitized adoo ilaalinaya columns-ka midaysan."""
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    ds = ds.map(_format_code_prompt, remove_columns=ds.column_names)
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds


def load_multi_domain(math_ratio: float = 0.8, n_total: int = 1000) -> Dataset:
    """
    Isku dar Math + Code iyadoo si ammaan ah loo cabbirayo tirada MBPP.
    GSM8K (weyn) + MBPP (yar) -> Unified Dataset.
    """
    # 1. Soo deji dhammaan xogta labada dhinac
    math_ds = load_math_dataset(split="train")
    code_ds = load_code_dataset(split="train")

    # 2. Xisaabi inta tusaale ee dhinac kasta laga qaadan karo iyadoon xadka la dhaafin
    target_code = min(len(code_ds), int(n_total * (1.0 - math_ratio)))
    target_math = min(len(math_ds), n_total - target_code)

    math_subset = math_ds.shuffle(seed=42).select(range(target_math))
    code_subset = code_ds.shuffle(seed=42).select(range(target_code))

    # 3. Isku dar labada qaybood
    combined = concatenate_datasets([math_subset, code_subset]).shuffle(seed=42)
    return combined