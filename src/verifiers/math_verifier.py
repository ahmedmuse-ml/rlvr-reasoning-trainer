import sympy
from sympy.parsing.sympy_parser import parse_expr

from src.utils.parser import CoTParser
from src.utils.timeout import time_limit, TimeoutError as SoftTimeout
from src.verifiers.base import VerifierInterface


class MathVerifier(VerifierInterface[str, str]):
    @staticmethod
    def _clean_str(text: str) -> str:
        return text.strip().lower().replace(",", "").replace("$", "")

    @classmethod
    def verify(cls, prediction: str, reference: str) -> bool:
        """
        Verify a mathematical model completion against the expected answer.

        The completion must contain an explicitly identified final answer
        using either the XML answer tag or boxed-answer format.
        """
        parsed = CoTParser.parse_completion(prediction)

        # Only accept an explicitly identified final answer.
        if parsed.final_answer is None:
            return False

        if parsed.extraction_source not in {"xml_tag", "boxed"}:
            return False

        clean_pred = cls._clean_str(parsed.final_answer)
        clean_reference = cls._clean_str(reference)

        if clean_pred == clean_reference:
            return True

        try:
            with time_limit(3):
                expr_pred = parse_expr(clean_pred)
                expr_reference = parse_expr(clean_reference)
                return bool(sympy.simplify(expr_pred - expr_reference) == 0)
        except (Exception, SoftTimeout):
            return False