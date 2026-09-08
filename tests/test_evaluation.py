from src.evaluation.evaluate import Evaluator

def test_evaluator_math_accuracy():
    evaluator = Evaluator()

    tasks = [
        {
            "completion": "<think>5+5=10</think><answer>10</answer>",
            "domain": "math",
            "answer": "10",
            "test_list": [],
        },
        {
            "completion": "<think>5+5=10</think><answer>9</answer>",
            "domain": "math",
            "answer": "10",
            "test_list": [],
        },
    ]

    results = evaluator.evaluate(None, tasks)

    assert results["accuracy"] == 0.5
    assert results["math_accuracy"] == 0.5
    assert results["code_accuracy"] == 0.0
    assert results["total"] == 2


def test_evaluator_code_accuracy():
    evaluator = Evaluator()

    tasks = [
        {
            "completion": (
                "<think>multiply</think>"
                "<answer>```python\n"
                "def multiply(a, b):\n"
                "    return a * b\n"
                "```</answer>"
            ),
            "domain": "code",
            "answer": "",
            "test_list": ["assert multiply(3, 4) == 12"],
        },
        {
            "completion": (
                "<think>wrong</think>"
                "<answer>```python\n"
                "def multiply(a, b):\n"
                "    return a + b\n"
                "```</answer>"
            ),
            "domain": "code",
            "answer": "",
            "test_list": ["assert multiply(3, 4) == 12"],
        },
    ]

    results = evaluator.evaluate(None, tasks)

    assert results["accuracy"] == 0.5
    assert results["math_accuracy"] == 0.0
    assert results["code_accuracy"] == 0.5
    assert results["total"] == 2


def test_evaluator_mixed_domains():
    evaluator = Evaluator()

    tasks = [
        {
            "completion": "<think>correct</think><answer>10</answer>",
            "domain": "math",
            "answer": "10",
            "test_list": [],
        },
        {
            "completion": (
                "<think>correct</think>"
                "<answer>```python\n"
                "def multiply(a, b):\n"
                "    return a * b\n"
                "```</answer>"
            ),
            "domain": "code",
            "answer": "",
            "test_list": ["assert multiply(3, 4) == 12"],
        },
    ]

    results = evaluator.evaluate(None, tasks)

    assert results["accuracy"] == 1.0
    assert results["math_accuracy"] == 1.0
    assert results["code_accuracy"] == 1.0
    assert results["total"] == 2


def test_evaluator_empty_tasks():
    evaluator = Evaluator()

    results = evaluator.evaluate(None, [])

    assert results == {
        "accuracy": 0.0,
        "math_accuracy": 0.0,
        "code_accuracy": 0.0,
        "total": 0,
    }