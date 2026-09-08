from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar


TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


class EvaluationInterface(ABC, Generic[TaskT, ResultT]):
    """
    Common contract for model evaluation.

    An evaluator runs a model on a collection of tasks and returns
    structured evaluation results.
    """

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        tasks: list[TaskT],
    ) -> ResultT:
        """
        Evaluate a model on the supplied tasks.

        Returns:
            Structured evaluation results.
        """
        raise NotImplementedError