# Step 25A — Pass database_path Through the Reward Manager

from typing import Any

from src.rewards.base import RewardInterface


class RewardManager:
    """
    Compose multiple reward components into a single TRL-compatible reward
    function.

    Each reward component must implement RewardInterface.
    """

    def __init__(
        self,
        components: list[RewardInterface],
    ):
        if not components:
            raise ValueError("At least one reward component is required.")

        self.components = components

    def __call__(
        self,
        completions: list[Any],
        **kwargs,
    ) -> list[float]:
        total_rewards = [0.0] * len(completions)

        context = {
            key: value
            for key, value in kwargs.items()
            if key in {
                "answer",
                "test_list",
                "domain",
                "database_sql",
                "database_path",
                "reference_sql",
            }
        }

        for component in self.components:
            rewards = component.compute(
                completions,
                context=context,
            )

            if len(rewards) != len(completions):
                raise ValueError(
                    f"{component.__class__.__name__} returned "
                    f"{len(rewards)} rewards for {len(completions)} completions."
                )

            total_rewards = [
                total + reward
                for total, reward in zip(total_rewards, rewards)
            ]

        return total_rewards