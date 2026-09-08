import re
from src.rewards.base import RewardInterface


class FormatReward(RewardInterface[str, None]):
    """
    Reward component for structured XML-style model responses.

    A completion receives +0.5 when it contains both:
        <think>...</think>
        <answer>...</answer>
    Otherwise it receives 0.0.
    """

    def __init__(self):
        self.pattern = re.compile(
            r"<think>.*?</think>\s*<answer>.*?</answer>",
            re.DOTALL | re.IGNORECASE,
        )

    def compute(
        self,
        completions: list[str],
        context: None = None,
    ) -> list[float]:
        rewards = []

        for completion in completions:
            text = str(completion).strip()

            if self.pattern.search(text):
                rewards.append(0.5)
            else:
                rewards.append(0.0)

        return rewards


def compute_format_reward(
    completions,
    **kwargs,
) -> list[float]:
    """
    TRL-compatible adapter for the FormatReward component.
    """
    return FormatReward().compute(
        [
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
            for completion in completions
        ]
    )