# Step 23A — Create the SQaLe Dataset Loader

from pathlib import Path

loader_path = Path(
    "/content/rlvr-reasoning-trainer/src/data/sqale_loader.py"
)

loader_code = r'''
# Step 23 — Implement Correct SQaLe Dataset Loader

import json
from pathlib import Path

from datasets import Dataset, load_dataset


SYSTEM_PROMPT = (
    "You are an SQL data analysis assistant. "
    "Generate a read-only SQL query that answers the user's question. "
    "Use only the tables and columns provided in the schema. "
    "Return your reasoning inside <think>...</think> tags "
    "and the SQL query inside <answer>...</answer> tags."
)


def _parse_json_field(value, default):
    """
    Parse a SQaLe JSON-encoded field.

    SQaLe stores fields such as relevant tables and execution results
    as JSON strings in the raw dataset.
    """
    if value is None:
        return default

    if isinstance(value, (list, dict)):
        return value

    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _format_sqale_task(
    row: dict,
    database_dir: Path,
) -> dict:
    """
    Convert one raw SQaLe row into the unified RLVR dataset structure.
    """

    question_id = str(row.get("question id") or "")
    schema_id = str(row.get("schema id") or "")
    question = str(row.get("question") or "")
    schema = str(row.get("Full schema") or "")
    reference_sql = str(row.get("sql statament") or "")
    difficulty = str(row.get("difficulty") or "")
    question_style = str(row.get("question style") or "")

    relevant_tables = _parse_json_field(
        row.get("relevant tables"),
        default=[],
    )

    execution_result = _parse_json_field(
        row.get("execution_result"),
        default=[],
    )

    if not question_id:
        raise ValueError("SQaLe row is missing 'question id'.")

    if not schema_id:
        raise ValueError(
            f"SQaLe question '{question_id}' is missing 'schema id'."
        )

    if not question:
        raise ValueError(
            f"SQaLe question '{question_id}' is missing question text."
        )

    if not schema:
        raise ValueError(
            f"SQaLe question '{question_id}' is missing 'Full schema'."
        )

    if not reference_sql:
        raise ValueError(
            f"SQaLe question '{question_id}' is missing gold SQL."
        )

    database_path = database_dir / f"{schema_id}.db"

    if not database_path.exists():
        raise FileNotFoundError(
            f"Materialized SQaLe database not found for schema "
            f"'{schema_id}': {database_path}"
        )

    return {
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Database schema:\n"
                    f"{schema}\n\n"
                    f"Question:\n"
                    f"{question}"
                ),
            },
        ],
        "answer": "",
        "test_list": [],
        "domain": "sql",
        "database_path": str(database_path),
        "reference_sql": reference_sql,
        "question_id": question_id,
        "schema_id": schema_id,
        "difficulty": difficulty,
        "question_style": question_style,
        "relevant_tables": relevant_tables,
        "execution_result": execution_result,
    }


def load_sqale_dataset(
    database_dir: str,
    max_samples: int | None = None,
    file_path: str = "trl-lab/SQaLe_2",
) -> Dataset:
    """
    Load SQaLe question-level data into the unified RLVR dataset format.

    Parameters
    ----------
    database_dir:
        Runtime directory containing materialized SQaLe SQLite databases.

    max_samples:
        Maximum number of SQaLe questions to load.

    file_path:
        Local SQaLe dataset path or Hugging Face dataset repository ID.

    Returns
    -------
    Dataset
        SQaLe questions in the unified RLVR dataset format.
    """

    database_dir_path = Path(database_dir)

    if not database_dir_path.exists():
        raise FileNotFoundError(
            f"SQaLe database directory does not exist: "
            f"{database_dir_path}"
        )

    dataset = load_dataset(
        file_path,
        split="train",
        streaming=True,
    )

    rows = []

    for row in dataset:
        rows.append(
            _format_sqale_task(
                row=row,
                database_dir=database_dir_path,
            )
        )

        if max_samples is not None and len(rows) >= max_samples:
            break

    if not rows:
        raise ValueError(
            "No SQaLe questions were loaded."
        )

    return Dataset.from_list(rows)
'''

loader_path.parent.mkdir(parents=True, exist_ok=True)
loader_path.write_text(loader_code)

print("Created:", loader_path)
print("Exists:", loader_path.exists())
print("Size:", loader_path.stat().st_size, "bytes")