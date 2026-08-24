from typing import List, Union, Optional
from src.verifiers.math_verifier import MathVerifier
from src.verifiers.code_verifier import CodeVerifier


def compute_accuracy_reward(
    completions: List[Union[str, List[dict]]],
    answer: Optional[List[str]] = None,
    test_list: Optional[List[List[str]]] = None,
    domain: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """
    Xisaabi abaalmarinta saxnaanta (Accuracy):
    - Math -> SymPy Symbolic Verification (+1.0 ama 0.0)
    - Code -> Sandboxed Test Execution (+1.0 ama 0.0)
    """
    rewards = []
    n = len(completions)

    answers = answer if answer is not None else [""] * n
    test_lists = test_list if test_list is not None else [[]] * n
    domains = domain if domain is not None else ["math"] * n

    for comp, ans, tests, dom in zip(completions, answers, test_lists, domains):
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)

        if dom == "code" or len(tests) > 0:
            is_correct = CodeVerifier.verify(text, tests)
        else:
            is_correct = MathVerifier.verify(text, ans)

        rewards.append(1.0 if is_correct else 0.0)

    return rewards