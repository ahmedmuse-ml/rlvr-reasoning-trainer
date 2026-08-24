import re
from typing import List, Union


def compute_format_reward(
    completions: List[Union[str, List[dict]]],
    **kwargs
) -> List[float]:
    """
    Reward u dhiirrigeliya XML-tags qaabaysan:
    +0.5 haddii uu leeyahay <think>...</think> iyo <answer>...</answer>
     0.0 haddii kale
    """
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
    rewards = []

    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        text_clean = text.strip()

        if pattern.search(text_clean):
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards