from src.utils.parser import CoTParser
from src.verifiers.base import VerifierInterface
from src.verifiers.sandbox_executor import SandboxExecutor


_executor = SandboxExecutor()


class CodeVerifier(VerifierInterface[str, list[str]]):
    @classmethod
    def verify(cls, prediction: str, reference: list[str]) -> bool:
        """
        Verify generated code by executing it against the supplied tests.

        Returns:
            True when the generated solution passes the test harness.
            False when no answer is extracted or execution fails.
        """
        parsed = CoTParser.parse_completion(prediction)
        code = parsed.final_answer

        if code is None:
            return False

        code = code.replace("```python", "").replace("```", "").strip()

        test_harness = "\n".join(reference)
        full_script = f"{code}\n\n{test_harness}\n"

        success, _ = _executor.run(full_script)
        return bool(success)