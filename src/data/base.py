from abc import ABC, abstractmethod
from typing import Generic, TypeVar


TaskT = TypeVar("TaskT")
PredictionT = TypeVar("PredictionT")


class TaskInterface(ABC, Generic[TaskT, PredictionT]):
    """
    Common contract for task definitions.

    A task provides the model input and task-specific reference data,
    and can expose the information required by its verifier.
    """

    @abstractmethod
    def get_prompt(self, task: TaskT) -> list[dict]:
        """
        Return the structured prompt used by the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_reference(self, task: TaskT) -> dict:
        """
        Return task-specific reference data required for verification.
        """
        raise NotImplementedError