from typing import Dict, Any, Optional

from datasets import Dataset, concatenate_datasets, load_dataset


def _format_math_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format a GSM8K example into the unified RLVR dataset structure."""
    raw_answer = example["answer"]
    clean_answer = (
        raw_answer.split("####")[-1].strip()
        if "####" in raw_answer
        else raw_answer.strip()
    )

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
            {
                "role": "user",
                "content": example["question"],
            },
        ],
        "answer": clean_answer,
        "test_list": [],
        "domain": "math",
    }


def _format_code_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format an MBPP example into the unified RLVR dataset structure."""
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
            {
                "role": "user",
                "content": task_description,
            },
        ],
        "answer": "",
        "test_list": example.get("test_list", []),
        "domain": "code",
    }


def load_math_dataset(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load and format the GSM8K dataset."""
    ds = load_dataset("openai/gsm8k", "main", split=split)

    if max_samples is not None:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))

    ds = ds.map(
        _format_math_prompt,
        remove_columns=ds.column_names,
    )

    return ds


def load_code_dataset(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load and format the sanitized MBPP dataset."""
    ds = load_dataset(
        "google-research-datasets/mbpp",
        "sanitized",
        split=split,
    )

    if max_samples is not None:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))

    ds = ds.map(
        _format_code_prompt,
        remove_columns=ds.column_names,
    )

    return ds


def load_multi_domain(
    math_ratio: float = 0.8,
    n_total: int = 1000,
    seed: int = 42,
) -> Dataset:
    """
    Build a reproducible mixed Math + Code dataset.

    The requested dataset size and domain ratio are enforced as closely
    as possible within the available dataset sizes.
    """
    if not 0.0 <= math_ratio <= 1.0:
        raise ValueError("math_ratio must be between 0.0 and 1.0.")

    if n_total <= 0:
        raise ValueError("n_total must be greater than 0.")

    target_math = int(round(n_total * math_ratio))
    target_code = n_total - target_math

    # Load only the required number of examples.
    math_ds = load_math_dataset(
        split="train",
        max_samples=target_math,
    )

    code_ds = load_code_dataset(
        split="train",
        max_samples=target_code,
    )

    # If a domain does not contain enough examples, use all available
    # examples from that domain and fill the remaining capacity from
    # the other domain.
    actual_math = len(math_ds)
    actual_code = len(code_ds)
    total_available = actual_math + actual_code

    if total_available < n_total:
        raise ValueError(
            f"Requested {n_total} samples, but only {total_available} "
            "samples are available across the selected datasets."
        )

    combined = concatenate_datasets(
        [math_ds, code_ds]
    ).shuffle(seed=seed)

    return combined