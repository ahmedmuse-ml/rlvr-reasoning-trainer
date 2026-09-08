from abc import ABC, abstractmethod
from typing import Generic, TypeVar


PredictionT = TypeVar("PredictionT")
ReferenceT = TypeVar("ReferenceT")


class VerifierInterface(ABC, Generic[PredictionT, ReferenceT]):
    """
    Common contract for verifier implementations.

    A verifier evaluates a model prediction against task-specific
    reference data and returns whether the prediction is correct.
    """

    @abstractmethod
    def verify(
        self,
        prediction: PredictionT,
        reference: ReferenceT,
    ) -> bool:
        """
        Verify a prediction against its task-specific reference.

        Returns:
            True when the prediction is correct, otherwise False.
        """
        raise NotImplementedError