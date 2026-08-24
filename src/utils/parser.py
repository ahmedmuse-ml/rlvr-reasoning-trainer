import re
from typing import Optional
from pydantic import BaseModel, Field


class ParsedCoT(BaseModel):
    reasoning: Optional[str] = None
    final_answer: Optional[str] = None
    has_valid_tags: bool = False
    is_truncated: bool = False
    extraction_source: str = Field(default="none")

    def __iter__(self):
        return iter((self.reasoning, self.final_answer))


class CoTParser:
    # Strict matching oo raba furitaan iyo xidhid buuxa
    THINK_REGEX = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    ANSWER_REGEX = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    @classmethod
    def extract_boxed(cls, text: str) -> Optional[str]:
        """Soo saar LaTeX \\boxed{...}. Haddii bracket-ku xidhmi waayo wuxuu soo celinayaa None."""
        idx = text.rfind(r"\boxed{")
        if idx == -1:
            return None

        start_idx = idx + len(r"\boxed{")
        depth = 1
        pos = start_idx

        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1

        if depth == 0:
            return text[start_idx:pos - 1].strip()
        
        return None

    @classmethod
    def parse_completion(cls, completion_text: str) -> ParsedCoT:
        if not completion_text or not isinstance(completion_text, str):
            return ParsedCoT()

        text = completion_text.strip()
        reasoning = None
        final_answer = None
        source = "none"

        # 1. Strict <think>...</think>
        think_match = cls.THINK_REGEX.search(text)
        if think_match:
            reasoning = think_match.group(1).strip()

        # 2. Strict <answer>...</answer>
        answer_match = cls.ANSWER_REGEX.search(text)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            source = "xml_tag"

        # 3. Fallback: LaTeX \\boxed{...}
        if not final_answer:
            boxed = cls.extract_boxed(text)
            if boxed:
                final_answer = boxed
                source = "boxed"

        # 4. Fallback: Lambarka ugu dambeeya qoraalka
        if not final_answer:
            nums = re.findall(r"[-+]?\d*\.?\d+", text)
            if nums:
                final_answer = nums[-1]
                source = "last_number"

        has_valid_tags = bool(
            "<think>" in text.lower() and "</think>" in text.lower() and 
            "<answer>" in text.lower() and "</answer>" in text.lower()
        )

        return ParsedCoT(
            reasoning=reasoning,
            final_answer=final_answer,
            has_valid_tags=has_valid_tags,
            is_truncated=bool("<think>" in text.lower() and "</think>" not in text.lower()),
            extraction_source=source,
        )