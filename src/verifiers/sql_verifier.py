import re
import sqlite3
from collections import Counter
from typing import Any

from src.utils.parser import CoTParser
from src.verifiers.base import VerifierInterface


class SQLVerifier(VerifierInterface[str, dict]):
    """
    Verify generated SQL by executing it against an isolated SQLite database.

    A prediction is correct when:
        1. A SQL query is extracted from the model completion.
        2. The query is read-only.
        3. The query executes successfully.
        4. Its result matches the trusted reference query result.

    The database is created in memory for every verification call.
    """

    _READ_ONLY_PATTERN = re.compile(
        r"^\s*(SELECT|WITH)\b",
        re.IGNORECASE,
    )

    _FORBIDDEN_PATTERN = re.compile(
        r"\b("
        r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|"
        r"ATTACH|DETACH|PRAGMA|VACUUM|REINDEX"
        r")\b",
        re.IGNORECASE,
    )

    @classmethod
    def _extract_sql(cls, prediction: str) -> str | None:
        """
        Extract SQL from the structured model completion.
        """
        parsed = CoTParser.parse_completion(prediction)

        if parsed.final_answer is None:
            return None

        sql = parsed.final_answer.strip()

        # Remove optional Markdown code fences.
        sql = sql.replace("```sql", "").replace("```SQL", "")
        sql = sql.replace("```", "").strip()

        return sql or None

    @classmethod
    def _is_read_only(cls, sql: str) -> bool:
        """
        Check whether generated SQL starts with an allowed read-only
        statement and contains no explicitly forbidden operations.
        """
        if not cls._READ_ONLY_PATTERN.match(sql):
            return False

        if cls._FORBIDDEN_PATTERN.search(sql):
            return False

        return True

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """
        Normalize SQLite values before result comparison.

        Floating-point values are rounded to reduce insignificant
        numerical representation differences.
        """
        if isinstance(value, float):
            return round(value, 10)

        return value

    @classmethod
    def _normalize_rows(
        cls,
        rows: list[tuple],
    ) -> Counter:
        """
        Convert query results into a row multiset.

        Row order is ignored because SQL does not guarantee ordering
        unless ORDER BY is explicitly requested. Duplicate rows remain
        significant.
        """
        normalized = []

        for row in rows:
            normalized.append(
                tuple(
                    cls._normalize_value(value)
                    for value in row
                )
            )

        return Counter(normalized)

    @staticmethod
    def _execute_query(
        connection: sqlite3.Connection,
        sql: str,
        timeout_seconds: float = 2.0,
    ) -> list[tuple]:
        """
        Execute a read-only query with a SQLite progress timeout.
        """
        max_steps = max(1, int(timeout_seconds * 1_000_000))
        steps = 0

        def progress_handler() -> int:
            nonlocal steps
            steps += 1
            return 1 if steps >= max_steps else 0

        connection.set_progress_handler(progress_handler, 1000)

        try:
            cursor = connection.execute(sql)
            return cursor.fetchall()
        finally:
            connection.set_progress_handler(None, 0)

    @classmethod
    def verify(
        cls,
        prediction: str,
        reference: dict,
    ) -> bool:
        """
        Verify generated SQL against a trusted reference query.

        Reference data must contain:
            database_sql: SQL statements creating and populating the database.
            reference_sql: Trusted SQL query defining the expected result.
        """
        if not isinstance(reference, dict):
            return False

        database_sql = reference.get("database_sql")
        reference_sql = reference.get("reference_sql")

        if not database_sql or not reference_sql:
            return False

        generated_sql = cls._extract_sql(prediction)

        if generated_sql is None:
            return False

        if not cls._is_read_only(generated_sql):
            return False

        try:
            connection = sqlite3.connect(":memory:")

            try:
                connection.executescript(database_sql)

                expected_rows = cls._execute_query(
                    connection,
                    reference_sql,
                )

                generated_rows = cls._execute_query(
                    connection,
                    generated_sql,
                )

                return (
                    cls._normalize_rows(expected_rows)
                    == cls._normalize_rows(generated_rows)
                )

            finally:
                connection.close()

        except (sqlite3.Error, TimeoutError, ValueError):
            return False