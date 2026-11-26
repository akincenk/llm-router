"""Pick a model per request.

Strategies:
- "small" / "large": always one model (baselines).
- "rules": an explainable difficulty score decides (length, numbers, reasoning cues, several questions, code).
- "cascade": the small model answers first and may reply ESCALATE; then the large model answers.
- In "rules" and "cascade", an empty reply from the small model also goes to the large model.
Any strategy can sit behind a SemanticCache.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from llm_router.cache import SemanticCache, numbers
from llm_router.client import Completion

SYSTEM = "Answer correctly and briefly. Give the final answer in the last line."
ESCALATE = "ESCALATE"
CASCADE_NOTE = f" If you are not sure your answer is correct, reply with exactly {ESCALATE} and nothing else."
CUES = (
    "why",
    "explain",
    "compare",
    "prove",
    "step by step",
    "trade-off",
    "tradeoff",
    "derive",
    "design",
    "analyse",
    "analyze",
    "how many ways",
    "if ",
    "each",
    "remaining",
    "after",
    "probability",
    "rate",
)


def difficulty(question: str) -> tuple[bool, list[str]]:
    """(is_hard, reasons). Two or more signals mean hard."""
    q = question.lower()
    reasons = []
    if len(q.split()) > 25:
        reasons.append("long")
    if len(numbers(q)) >= 3:
        reasons.append("3+ numbers")
    cues = [c.strip() for c in CUES if c in q]
    reasons += [f"cue:{c}" for c in cues[:2]]
    if q.count("?") > 1 or len(re.findall(r"[.!?](\s|$)", q)) > 2:
        reasons.append("several sentences")
    if "```" in q or "def " in q:
        reasons += ["code", "code"]
    return len(reasons) >= 2, reasons


@dataclass
class Answer:
    text: str
    model: str  # "cache" when served from the cache
    seconds: float = 0.0
    tokens_small: int = 0
    tokens_large: int = 0
    escalated: bool = False
    reasons: list[str] = field(default_factory=list)


class Router:
    def __init__(
        self,
        chat: Callable[[str, list[dict]], Completion],
        small: str,
        large: str,
        strategy: str = "rules",
        cache: SemanticCache | None = None,
    ):
        if strategy not in ("small", "large", "rules", "cascade"):
            raise ValueError("strategy must be small, large, rules or cascade")
        self.chat, self.small, self.large, self.strategy, self.cache = chat, small, large, strategy, cache

    def _ask(self, model: str, question: str, note: str = "") -> Completion:
        return self.chat(model, [{"role": "system", "content": SYSTEM + note}, {"role": "user", "content": question}])

    def ask(self, question: str) -> Answer:
        if self.cache is not None and (hit := self.cache.get(question)):
            return Answer(hit[0], "cache", reasons=[f"cache similarity {hit[1]:.3f}"])
        hard, reasons = difficulty(question)
        if self.strategy == "large" or (self.strategy == "rules" and hard):
            c = self._ask(self.large, question)
            ans = Answer(c.text, self.large, c.seconds, tokens_large=c.tokens, reasons=reasons)
        elif self.strategy == "small":
            c = self._ask(self.small, question)
            ans = Answer(c.text, self.small, c.seconds, tokens_small=c.tokens, reasons=reasons)
        else:  # rules (easy question) or cascade: the small model first, the large one if it gives up
            first = self._ask(self.small, question, CASCADE_NOTE if self.strategy == "cascade" else "")
            text = first.text.strip()
            # An empty reply (for example a thinking model that used its whole token budget) also escalates.
            if not text or text.upper().startswith(ESCALATE):
                second = self._ask(self.large, question)
                why = "small model escalated" if text else "small model gave no answer"
                ans = Answer(
                    second.text, self.large, first.seconds + second.seconds, first.tokens, second.tokens, True, [why]
                )
            else:
                ans = Answer(first.text, self.small, first.seconds, tokens_small=first.tokens, reasons=reasons)
        if self.cache is not None and ans.text.strip():  # an empty cache is falsy (len 0)
            self.cache.put(question, ans.text)
        return ans
