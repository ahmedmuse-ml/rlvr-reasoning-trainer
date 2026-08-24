from typing import List, Union


def compute_length_reward(
    completions: List[Union[str, List[dict]]],
    max_reasonable_length: int = 500,
    penalty_scale: float = 0.001,
    **kwargs
) -> List[float]:
    """
    Ka hortag reward-hacking (verbosity):
    Ciqaab dhibco taban (-penalty) haddii tirada ereyadu ka badato 500 eray.
    """
    rewards = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        words_count = len(text.split())
        excess = max(0, words_count - max_reasonable_length)
        rewards.append(-penalty_scale * excess)

    return rewards