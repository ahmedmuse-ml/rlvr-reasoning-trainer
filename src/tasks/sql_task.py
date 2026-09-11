from dataclasses import dataclass

from src.data.base import TaskInterface


@dataclass
class SQLTask:
    """
    Represents one SQL question-answering task.

    Attributes:
        question: Natural-language question the model must answer.
        schema: SQL schema shown to the model.
        database_sql: SQL statements used to construct the evaluation database.
        reference_sql: Trusted SQL query used to determine the expected result.
    """

    question: str
    schema: str
    database_sql: str
    reference_sql: str


class SQLTaskInterface(TaskInterface[SQLTask, str]):
    """
    Task interface for SQL generation and verification.

    The model receives only the question and database schema.
    The database setup and reference SQL remain evaluation-side data.
    """

    def get_prompt(self, task: SQLTask) -> list[dict]:
        """
        Build the structured prompt presented to the model.
        """
        return [
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
        ]

    def get_reference(self, task: SQLTask) -> dict:
        """
        Return evaluation-side information required by the SQL verifier.

        The reference SQL and database setup are not exposed in the model prompt.
        """
        return {
            "database_sql": task.database_sql,
            "reference_sql": task.reference_sql,
        }