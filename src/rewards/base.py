from abc import ABC, abstractmethod
from typing import Generic, TypeVar


CompletionT = TypeVar("CompletionT")
ContextT = TypeVar("ContextT")


class RewardInterface(ABC, Generic[CompletionT, ContextT]):
    """
    Common contract for reward components.

    A reward component evaluates one or more model completions using
    optional task-specific context and returns a scalar reward for each
    completion.
    """

    @abstractmethod
    def compute(
        self,
        completions: list[CompletionT],
        context: ContextT | None = None,
    ) -> list[float]:
        """
        Compute one reward value for each completion.

        Returns:
            A list of scalar rewards with the same length as completions.
        """
        raise NotImplementedError