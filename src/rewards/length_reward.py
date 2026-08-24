from typing import List, Union


def compute_length_reward(
    completions: List[Union[str, List[dict]]],
    max_reasonable_length: int = 600,
    penalty_scale: float = 0.001,
    **kwargs
) -> List[float]:
    """
    Ka hortag reward hacking: model-ku ma helayo faa'iido dheeraad ah
    isagoo isu dherejinaya completion-ka. Ciqaab tokens ka badan xadka.
    """
    rewards = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        n_tokens = len(text.split())
        excess = max(0, n_tokens - max_reasonable_length)
        rewards.append(-penalty_scale * excess)
    return rewards