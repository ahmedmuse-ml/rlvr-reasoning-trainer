from transformers import TrainerCallback
import statistics


class RewardHackingMonitor(TrainerCallback):
    """
    Monitor the relationship between completion length and reward.

    A strong positive correlation can be a warning sign that the model
    may be receiving higher rewards primarily for producing longer
    completions. Correlation alone does not prove reward hacking.
    """

    def __init__(self, threshold: float = 0.5, window_size: int = 20):
        self.threshold = threshold
        self.window_size = window_size
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control

        reward = logs.get("reward")
        completion_length = logs.get("completions/mean_length")

        if reward is not None and completion_length is not None:
            self.history.append(
                (float(completion_length), float(reward))
            )

        if len(self.history) >= self.window_size:
            recent = self.history[-self.window_size:]

            lengths = [item[0] for item in recent]
            rewards = [item[1] for item in recent]

            try:
                correlation = statistics.correlation(lengths, rewards)

                if correlation > self.threshold:
                    print(
                        "[REWARD-MONITOR] Warning: "
                        f"completion-length/reward correlation = "
                        f"{correlation:.2f}"
                    )

            except statistics.StatisticsError:
                # Correlation cannot be calculated when either variable
                # has no variation.
                pass

        return control