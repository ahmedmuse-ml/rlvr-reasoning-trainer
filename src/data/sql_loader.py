from datasets import Dataset

from data.sql.sales_tasks import SQL_TASKS


def _format_sql_task(task) -> dict:
    """
    Convert an SQLTask into the unified RLVR dataset structure.
    """
    return {
        "prompt": [
            {
                "role": "system",
                "content": (
                    "You are an SQL data analysis assistant. "
                    "Generate a read-only SQL query that answers the user's question. "
                    "Use only the tables and columns provided in the schema. "
                    "Return your reasoning inside <think>...</think> tags "
                    "and the SQL query inside <answer>...</answer> tags."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Database schema:\n"
                    f"{task.schema}\n\n"
                    f"Question:\n"
                    f"{task.question}"
                ),
            },
        ],
        "answer": "",
        "test_list": [],
        "domain": "sql",
        "database_sql": task.database_sql,
        "reference_sql": task.reference_sql,
    }


def load_sql_dataset(
    max_samples: int | None = None,
) -> Dataset:
    """
    Load the SQL application dataset into the unified RLVR format.
    """
    tasks = SQL_TASKS

    if max_samples is not None:
        max_samples = min(max_samples, len(tasks))
        tasks = tasks[:max_samples]

    return Dataset.from_list(
        [_format_sql_task(task) for task in tasks]
    )