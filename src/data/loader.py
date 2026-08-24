from datasets import load_dataset, Dataset, concatenate_datasets


def load_math_dataset(split: str = "train") -> Dataset:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return ds.map(_format_math_prompt)


def load_code_dataset(split: str = "train") -> Dataset:
    ds = load_dataset("mbpp", split=split)
    return ds.map(_format_code_prompt)


def _format_math_prompt(example: dict) -> dict:
    example["prompt"] = [
        {"role": "system", "content": "Solve step by step. Respond in the format: <think>...</think><answer>...</answer>"},
        {"role": "user", "content": example["question"]},
    ]
    example["answer"] = example["answer"].split("####")[-1].strip()
    return example


def _format_code_prompt(example: dict) -> dict:
    example["prompt"] = [
        {"role": "system", "content": "Write a Python function. Respond in the format: <think>...</think><answer>```python\n...\n```</answer>"},
        {"role": "user", "content": example["text"]},
    ]
    example["test_list"] = example.get("test_list", [])
    return example


def load_multi_domain(math_ratio: float = 0.5, n_total: int = 2000) -> Dataset:
    """Isku dar Math + Code, curriculum-ready."""
    n_math = int(n_total * math_ratio)
    n_code = n_total - n_math
    math_ds = load_math_dataset().shuffle(seed=42).select(range(n_math))
    code_ds = load_code_dataset().shuffle(seed=42).select(range(n_code))
    return concatenate_datasets([math_ds, code_ds])