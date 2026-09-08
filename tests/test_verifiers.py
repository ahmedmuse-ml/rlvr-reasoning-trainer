from src.verifiers.math_verifier import MathVerifier
from src.verifiers.code_verifier import CodeVerifier


# -------------------------
# Math verifier tests
# -------------------------

def test_math_correct_xml_answer():
    prediction = (
        "<think>10 is the calculated result.</think>"
        "<answer>10</answer>"
    )

    assert MathVerifier.verify(prediction, "10") is True


def test_math_incorrect_answer():
    prediction = (
        "<think>The calculation gives 12.</think>"
        "<answer>12</answer>"
    )

    assert MathVerifier.verify(prediction, "10") is False


def test_math_boxed_answer():
    prediction = (
        "<think>Calculate the result.</think>"
        "The final result is \\boxed{10}"
    )

    assert MathVerifier.verify(prediction, "10") is True


def test_math_missing_answer():
    prediction = "<think>The answer should be 10."

    assert MathVerifier.verify(prediction, "10") is False


# -------------------------
# Code verifier tests
# -------------------------

def test_code_correct_solution():
    prediction = (
        "<answer>```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```</answer>"
    )

    tests = ["assert add(2, 3) == 5"]

    assert CodeVerifier.verify(prediction, tests) is True


def test_code_incorrect_solution():
    prediction = (
        "<answer>```python\n"
        "def add(a, b):\n"
        "    return a - b\n"
        "```</answer>"
    )

    tests = ["assert add(2, 3) == 5"]

    assert CodeVerifier.verify(prediction, tests) is False


def test_code_missing_answer():
    prediction = (
        "<think>Define a function that adds two numbers.</think>"
    )

    tests = ["assert add(2, 3) == 5"]

    assert CodeVerifier.verify(prediction, tests) is False
