from src.rewards.accuracy_reward import compute_accuracy_reward
from src.rewards.format_reward import compute_format_reward
from src.rewards.length_reward import compute_length_reward


def test_math_accuracy_reward():
    completions = ["<think>5+5=10</think><answer>10</answer>", "<think>khalad</think><answer>9</answer>"]
    answers = ["10", "10"]
    domains = ["math", "math"]

    rewards = compute_accuracy_reward(completions=completions, answer=answers, domain=domains)
    assert rewards == [1.0, 0.0]


def test_code_accuracy_reward():
    correct_code = "<think>ku dar</think><answer>```python\ndef multiply(a, b):\n    return a * b\n```</answer>"
    wrong_code = "<think>khalad</think><answer>```python\ndef multiply(a, b):\n    return a + b\n```</answer>"

    completions = [correct_code, wrong_code]
    test_lists = [["assert multiply(3, 4) == 12"], ["assert multiply(3, 4) == 12"]]
    domains = ["code", "code"]

    rewards = compute_accuracy_reward(completions=completions, test_list=test_lists, domain=domains)
    assert rewards == [1.0, 0.0]


def test_format_reward_tag_detection():
    valid = "<think>sababayn</think><answer>42</answer>"
    invalid = "Kaliya jawaabta waa 42"

    rewards = compute_format_reward(completions=[valid, invalid])
    assert rewards == [0.5, 0.0]


def test_length_reward_penalty():
    short_text = "<think>waa gaaban yahay</think><answer>1</answer>"
    long_text = "erey " * 600

    rewards = compute_length_reward(completions=[short_text, long_text], max_reasonable_length=500)
    assert rewards[0] == 0.0
    assert rewards[1] < 0.0