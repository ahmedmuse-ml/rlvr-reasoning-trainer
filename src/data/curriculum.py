from collections import defaultdict
from datasets import Dataset


class DifficultyTracker:
    """Ku xasuuso pass-rate-ka prompt kasta si loo sameeyo curriculum (fudud -> adag)."""

    def __init__(self, ema_alpha: float = 0.1):
        self.pass_rate: dict[str, float] = defaultdict(lambda: 0.5)
        self.alpha = ema_alpha

    def update(self, prompt_id: str, rewards: list[float]):
        rate = sum(1 for r in rewards if r >= 1.0) / max(len(rewards), 1)
        prev = self.pass_rate[prompt_id]
        self.pass_rate[prompt_id] = (1 - self.alpha) * prev + self.alpha * rate

    def filter_by_difficulty(self, dataset: Dataset, min_rate: float, max_rate: float) -> Dataset:
        """Kaliya prompts-ka aan si buuxda loo bartay (0 < pass_rate < 1) — informative gradients."""
        def _keep(example, idx):
            pid = str(idx)
            rate = self.pass_rate[pid]
            return min_rate <= rate <= max_rate
        return dataset.filter(_keep, with_indices=True)