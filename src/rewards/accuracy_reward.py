from typing import Optional

from src.rewards.base import RewardInterface
from src.verifiers.code_verifier import CodeVerifier
from src.verifiers.math_verifier import MathVerifier
from src.verifiers.sql_verifier import SQLVerifier


class AccuracyReward(RewardInterface[str, dict]):
    """
    Reward component for task correctness.

    Math tasks are verified against an expected answer.
    Code tasks are verified by executing the generated solution
    against the supplied test harness.
    SQL tasks are verified by executing the generated query
    against an isolated database and comparing its result with
    a trusted reference query.

    Correct completions receive 1.0.
    Incorrect completions receive 0.0.
    """

    @staticmethod
    def _extract_text(completion: str | list[dict]) -> str:
        if isinstance(completion, list):
            return completion[0]["content"]

        return str(completion)

    def compute(
        self,
        completions: list[str | list[dict]],
        context: Optional[dict] = None,
    ) -> list[float]:
        context = context or {}

        answers = context.get("answer")
        test_list = context.get("test_list")
        domains = context.get("domain")
        database_sql = context.get("database_sql")
        reference_sql = context.get("reference_sql")

        n = len(completions)

        answers = answers if answers is not None else [""] * n
        test_list = (
            test_list
            if test_list is not None
            else [[] for _ in range(n)]
        )
        domains = (
            domains
            if domains is not None
            else ["math"] * n
        )
        database_sql = (
            database_sql
            if database_sql is not None
            else [""] * n
        )
        reference_sql = (
            reference_sql
            if reference_sql is not None
            else [""] * n
        )

        rewards = []

        for completion, answer, tests, domain, db_sql, ref_sql in zip(
            completions,
            answers,
            test_list,
            domains,
            database_sql,
            reference_sql,
        ):
            text = self._extract_text(completion)

            if domain == "sql":
                is_correct = SQLVerifier.verify(
                    text,
                    {
                        "database_sql": db_sql,
                        "reference_sql": ref_sql,
                    },
                )
            elif domain == "code" or len(tests) > 0:
                is_correct = CodeVerifier.verify(
                    text,
                    tests,
                )
            else:
                is_correct = MathVerifier.verify(
                    text,
                    answer,
                )

            rewards.append(1.0 if is_correct else 0.0)

        return rewards


def compute_accuracy_reward(
    completions,
    answer=None,
    test_list=None,
    domain=None,
    database_sql=None,
    reference_sql=None,
    **kwargs,
) -> list[float]:
    """
    TRL-compatible adapter for the AccuracyReward component.
    """
    return AccuracyReward().compute(
        completions,
        context={
            "answer": answer,
            "test_list": test_list,
            "domain": domain,
            "database_sql": database_sql,
            "reference_sql": reference_sql,
        },
    )