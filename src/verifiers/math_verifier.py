import sympy
from sympy.parsing.sympy_parser import parse_expr
from src.utils.parser import CoTParser
from src.utils.timeout import time_limit, TimeoutError as SoftTimeout


class MathVerifier:
    @staticmethod
    def _clean_str(text: str) -> str:
        return text.strip().lower().replace(",", "").replace("$", "")

    @classmethod
    def verify(cls, prediction_raw: str, ground_truth: str) -> bool:
        parsed = CoTParser.parse_completion(prediction_raw)
        if parsed.final_answer is None:
            return False

        clean_pred = cls._clean_str(parsed.final_answer)
        clean_gt = cls._clean_str(ground_truth)

        if clean_pred == clean_gt:
            return True

        try:
            with time_limit(3):
                expr_pred = parse_expr(clean_pred)
                expr_gt = parse_expr(clean_gt)
                return bool(sympy.simplify(expr_pred - expr_gt) == 0)
        except (Exception, SoftTimeout):
            return False