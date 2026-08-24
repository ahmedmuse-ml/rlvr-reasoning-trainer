from src.verifiers.code_verifier import CodeVerifier

def test_correct_solution():
    pred = "<answer>```python\ndef add(a, b):\n    return a + b\n```</answer>"
    tests = ["assert add(2, 3) == 5"]
    assert CodeVerifier.verify(pred, tests) is True

def test_infinite_loop_times_out():
    pred = "<answer>```python\nwhile True: pass\n```</answer>"
    assert CodeVerifier.verify(pred, []) is False