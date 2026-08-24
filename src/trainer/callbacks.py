from transformers import TrainerCallback
import statistics


class RewardHackingMonitor(TrainerCallback):
    """Digniin haddii reward-length correlation-ku kordho — calaamad reward hacking ah."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "reward" in logs and "completion_length" in logs:
            self.history.append((logs["completion_length"], logs["reward"]))

        if len(self.history) >= 20:
            lengths = [h[0] for h in self.history[-20:]]
            rewards = [h[1] for h in self.history[-20:]]
            try:
                corr = statistics.correlation(lengths, rewards)
                if corr > self.threshold:
                    print(f"⚠️  Reward-hacking warning: length↔reward correlation = {corr:.2f}")
            except statistics.StatisticsError:
                pass
        return control