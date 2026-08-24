from transformers import TrainerCallback
from src.data.curriculum import DifficultyTracker


class CurriculumCallback(TrainerCallback):
    """Cusboonaysii pass-rate tracker-ka reward kasta ka dib; ku filter dataset-ka
    marka epoch/step gaadho xadka la qeexay (progressive difficulty)."""

    def __init__(self, tracker: DifficultyTracker):
        self.tracker = tracker

    def on_step_end(self, args, state, control, **kwargs):
        rewards = kwargs.get("rewards")
        prompt_ids = kwargs.get("prompt_ids")
        if rewards and prompt_ids:
            for pid, r_group in zip(prompt_ids, rewards):
                self.tracker.update(pid, r_group)
        return control