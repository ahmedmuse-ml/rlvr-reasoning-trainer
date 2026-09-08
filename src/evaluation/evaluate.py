from typing import Any
from src.evaluation.base import EvaluationInterface
from src.verifiers.code_verifier import CodeVerifier
from src.verifiers.math_verifier import MathVerifier


class Evaluator(EvaluationInterface):
    """
    Concrete evaluator for verifier-based model evaluation.

    The evaluator receives generated completions together with their
    task references and computes domain-specific correctness metrics.
    """

    def evaluate(
        self,
        model: Any,
        tasks: list[dict],
    ) -> dict:
        """
        Evaluate supplied model outputs against task references.

        Each task must contain:
            completion: generated model output
            domain: task domain
            answer: expected math answer
            test_list: code test harness

        Returns:
            Overall and per-domain accuracy metrics.
        """
        if not tasks:
            return {
                "accuracy": 0.0,
                "math_accuracy": 0.0,
                "code_accuracy": 0.0,
                "total": 0,
            }

        correct = 0
        math_total = 0
        math_correct = 0
        code_total = 0
        code_correct = 0

        for task in tasks:
            completion = task.get("completion", "")
            domain = task.get("domain", "math")

            if domain == "code":
                code_total += 1
                is_correct = CodeVerifier.verify(
                    completion,
                    task.get("test_list", []),
                )

                if is_correct:
                    code_correct += 1
                    correct += 1

            else:
                math_total += 1
                is_correct = MathVerifier.verify(
                    completion,
                    task.get("answer", ""),
                )

                if is_correct:
                    math_correct += 1
                    correct += 1

        total = len(tasks)

        return {
            "accuracy": correct / total,
            "math_accuracy": (
                math_correct / math_total if math_total else 0.0
            ),
            "code_accuracy": (
                code_correct / code_total if code_total else 0.0
            ),
            "total": total,
        }