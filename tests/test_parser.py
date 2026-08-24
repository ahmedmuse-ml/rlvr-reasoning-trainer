from src.utils.parser import CoTParser


def test_full_xml_extraction():
    text = "<think>10 + 5 = 15</think><answer>15</answer>"
    reasoning, answer = CoTParser.parse_completion(text)
    assert reasoning == "10 + 5 = 15"
    assert answer == "15"


def test_nested_boxed():
    text = "Answer: \\boxed{\\frac{1}{2}}"
    boxed = CoTParser.extract_boxed(text)
    assert boxed == "\\frac{1}{2}"


def test_simple_boxed_fallback():
    text = "Sidaas darteed waxaan helay \\boxed{42}"
    reasoning, answer = CoTParser.parse_completion(text)
    assert answer == "42"


def test_number_fallback_when_no_tags():
    text = "Total-ku wuxuu noqonayaa 15 markaa xisaabtu waa saxan"
    reasoning, answer = CoTParser.parse_completion(text)
    assert answer == "15"
    assert reasoning is None


def test_malformed_missing_closing_tag():
    text = "<think>Waan xisaabinayaa<answer>7</answer>"
    reasoning, answer = CoTParser.parse_completion(text)
    # THINK_PATTERN ma helayo closing </think>, marka reasoning waa None
    assert reasoning is None
    assert answer == "7"


def test_no_answer_found_at_all():
    text = "Waxaan ka fikirayaa su'aasha laakiin ma jiro jawaab cad"
    reasoning, answer = CoTParser.parse_completion(text)
    assert answer is None


def test_unclosed_boxed_returns_none():
    text = "\\boxed{1/2"
    assert CoTParser.extract_boxed(text) is None