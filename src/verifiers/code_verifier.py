from src.utils.parser import CoTParser
from src.verifiers.sandbox_executor import SandboxExecutor

_executor = SandboxExecutor()


class CodeVerifier:
    @classmethod
    def verify(cls, prediction_raw: str, test_list: list[str]) -> bool:
        parsed = CoTParser.parse_completion(prediction_raw)
        code = parsed.final_answer
        if code is None:
            return False

        code = code.replace("```python", "").replace("```", "").strip()
        test_harness = "\n".join(test_list)
        full_script = f"{code}\n\n{test_harness}\n"

        success, _ = _executor.run(full_script)
        return success