import json
from src.data.loader import load_math_dataset, load_code_dataset
from src.verifiers.math_verifier import MathVerifier
from src.verifiers.code_verifier import CodeVerifier

def evaluate(model_path: str) -> dict:
    # TODO: load model_path via vLLM, generate completions, run verifiers
    results = {"gsm8k_accuracy": 0.0, "humaneval_pass@1": 0.0}
    with open("reports/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    evaluate("outputs/grpo-run-01/final")