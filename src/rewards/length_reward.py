import math

from src.rewards.base import RewardInterface


class LengthReward(RewardInterface[str, None]):
    """
    Reward component that applies smooth length regularization.

    Completions at or below the threshold receive no penalty.
    Longer completions receive a smooth cosine penalty capped at
    the configured maximum penalty.
    """

    def __init__(
        self,
        threshold_length: int = 350,
        max_length: int = 512,
        max_penalty: float = 0.2,
    ):
        self.threshold_length = threshold_length
        self.max_length = max_length
        self.max_penalty = max_penalty

    def compute(
        self,
        completions: list[str],
        context: None = None,
    ) -> list[float]:
        threshold = self.threshold_length
        max_len = max(self.max_length, threshold + 1)

        rewards = []

        for completion in completions:
            token_count = len(str(completion).split())

            if token_count <= threshold:
                rewards.append(0.0)
            elif token_count >= max_len:
                rewards.append(-float(self.max_penalty))
            else:
                progress = (token_count - threshold) / (max_len - threshold)
                cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
                penalty = -self.max_penalty * cosine_factor
                rewards.append(float(penalty))

        return rewards


def compute_length_reward(
    completions,
    threshold_length: int = 350,
    max_length: int = 512,
    max_penalty: float = 0.2,
    **kwargs,
) -> list[float]:
    """
    TRL-compatible adapter for the LengthReward component.

    Supports the legacy max_reasonable_length keyword used by
    existing callers and tests.
    """
    threshold = kwargs.get("max_reasonable_length", threshold_length)

    return LengthReward(
        threshold_length=threshold,
        max_length=max_length,
        max_penalty=max_penalty,
    ).compute(
        [
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
            for completion in completions
        ]
    )