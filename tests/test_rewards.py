from src.rewards.accuracy_reward import AccuracyReward, compute_accuracy_reward
from src.rewards.format_reward import FormatReward, compute_format_reward
from src.rewards.length_reward import LengthReward, compute_length_reward
from src.rewards.manager import RewardManager


def test_math_accuracy_reward():
    completions = [
        "<think>5+5=10</think><answer>10</answer>",
        "<think>khalad</think><answer>9</answer>",
    ]
    answers = ["10", "10"]
    domains = ["math", "math"]

    rewards = compute_accuracy_reward(
        completions=completions,
        answer=answers,
        domain=domains,
    )

    assert rewards == [1.0, 0.0]


def test_math_accuracy_rejects_number_only_in_reasoning():
    completion = "<think>The answer should be 10."

    rewards = compute_accuracy_reward(
        completions=[completion],
        answer=["10"],
        domain=["math"],
    )

    assert rewards == [0.0]


def test_code_accuracy_reward():
    correct_code = (
        "<think>ku dar</think>"
        "<answer>```python\n"
        "def multiply(a, b):\n"
        "    return a * b\n"
        "```</answer>"
    )

    wrong_code = (
        "<think>khalad</think>"
        "<answer>```python\n"
        "def multiply(a, b):\n"
        "    return a + b\n"
        "```</answer>"
    )

    completions = [correct_code, wrong_code]

    test_lists = [
        ["assert multiply(3, 4) == 12"],
        ["assert multiply(3, 4) == 12"],
    ]

    domains = ["code", "code"]

    rewards = compute_accuracy_reward(
        completions=completions,
        test_list=test_lists,
        domain=domains,
    )

    assert rewards == [1.0, 0.0]


def test_format_reward_tag_detection():
    valid = "<think>sababayn</think><answer>42</answer>"
    invalid = "Kaliya jawaabta waa 42"

    rewards = compute_format_reward(
        completions=[valid, invalid]
    )

    assert rewards == [0.5, 0.0]


def test_length_reward_penalty():
    short_text = "<think>waa gaaban yahay</think><answer>1</answer>"
    long_text = "erey " * 600

    rewards = compute_length_reward(
        completions=[short_text, long_text],
        max_reasonable_length=500,
    )

    assert rewards[0] == 0.0
    assert rewards[1] < 0.0


def test_accuracy_reward_component():
    reward = AccuracyReward()

    rewards = reward.compute(
        ["<think>5+5=10</think><answer>10</answer>"],
        context={
            "answer": ["10"],
            "test_list": [[]],
            "domain": ["math"],
        },
    )

    assert rewards == [1.0]


def test_format_reward_component():
    reward = FormatReward()

    rewards = reward.compute(
        ["<think>reasoning</think><answer>42</answer>"]
    )

    assert rewards == [0.5]


def test_length_reward_component():
    reward = LengthReward(
        threshold_length=5,
        max_length=10,
        max_penalty=0.2,
    )

    rewards = reward.compute(
        [
            "one two three",
            "one two three four five six seven eight nine ten eleven",
        ]
    )

    assert rewards[0] == 0.0
    assert rewards[1] == -0.2


def test_reward_manager_combines_components():
    manager = RewardManager(
        components=[
            AccuracyReward(),
            FormatReward(),
            LengthReward(),
        ]
    )

    rewards = manager(
        ["<think>5+5=10</think><answer>10</answer>"],
        answer=["10"],
        test_list=[[]],
        domain=["math"],
    )

    assert len(rewards) == 1
    assert rewards[0] == 1.5


def test_reward_manager_rejects_wrong_reward_length():
    class BadReward:
        def compute(self, completions, context=None):
            return []

    manager = RewardManager(components=[BadReward()])

    try:
        manager(["completion"])
        assert False
    except ValueError as exc:
        assert "returned 0 rewards for 1 completions" in str(exc)