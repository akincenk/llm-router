"""Pick a model per request.

Strategies:
- "small" / "large": always one model (baselines).
- "rules": an explainable difficulty score decides (length, numbers, reasoning cues, several questions, code).
- "cascade": the small model answers first and may reply ESCALATE; then the large model answers.
- "learned": a model trained on past answers predicts from the question's embedding whether the small model will
  be right (see learned.py); below the threshold the question goes to the large model.
- "consistency": the small model answers twice (the second time sampled); if the two final answers differ, the
  large model answers instead. Disagreement is a cheap sign of uncertainty that needs no training.
- In every strategy except the baselines, an empty reply from the small model also goes to the large model.
Any strategy can sit behind a SemanticCache.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from llm_router.cache import SemanticCache, numbers
from llm_router.client import Completion

SYSTEM = "Answer correctly and briefly. End with a last line of the form 'Answer: <final answer>'."
ESCALATE = "ESCALATE"
STRATEGIES = ("small", "large", "rules", "cascade", "learned", "consistency")
SAMPLE_TEMPERATURE = 0.8
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


def final_answer(text: str) -> str:
    """From the last non-empty line (commas removed): the first number after "Answer:" when the line has one
    (so "Answer: 180,000 m (180 km)" gives 180000), else the last number, else the line in lower case."""
    lines = [line for line in text.splitlines() if line.strip()]
    last = lines[-1].replace(",", "") if lines else ""
    tagged = re.split(r"answer\s*[:：]", last, flags=re.IGNORECASE)
    nums = re.findall(r"-?\d+(?:\.\d+)?", tagged[-1]) if len(tagged) > 1 else []
    nums = nums[:1] or re.findall(r"-?\d+(?:\.\d+)?", last)[-1:]
    if nums:
        v = float(nums[0])
        return str(int(v)) if v.is_integer() else str(v)
    return last.strip().lower()


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
        predictor: Callable[[str], float] | None = None,
        threshold: float = 0.5,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
        if strategy == "learned" and predictor is None:
            raise ValueError("the learned strategy needs a predictor")
        self.chat, self.small, self.large, self.strategy, self.cache = chat, small, large, strategy, cache
        self.predictor, self.threshold = predictor, threshold

    def _ask(self, model: str, question: str, note: str = "", temperature: float = 0) -> Completion:
        messages = [{"role": "system", "content": SYSTEM + note}, {"role": "user", "content": question}]
        return self.chat(model, messages, **({"temperature": temperature} if temperature else {}))

    def _route(self, question: str) -> tuple[bool, list[str]]:
        if self.strategy == "learned":
            p = self.predictor(question)
            return p < self.threshold, [f"p(small right)={p:.2f}"]
        return difficulty(question)

    def ask(self, question: str) -> Answer:
        if self.cache is not None and (hit := self.cache.get(question)):
            return Answer(hit[0], "cache", reasons=[f"cache similarity {hit[1]:.3f}"])
        hard, reasons = self._route(question)
        if self.strategy == "large" or (self.strategy in ("rules", "learned") and hard):
            c = self._ask(self.large, question)
            ans = Answer(c.text, self.large, c.seconds, tokens_large=c.tokens, reasons=reasons)
        elif self.strategy == "small":
            c = self._ask(self.small, question)
            ans = Answer(c.text, self.small, c.seconds, tokens_small=c.tokens, reasons=reasons)
        else:  # an easy question, cascade or consistency: the small model first, the large one if in doubt
            first = self._ask(self.small, question, CASCADE_NOTE if self.strategy == "cascade" else "")
            text = first.text.strip()
            disagree = False
            if self.strategy == "consistency" and text:
                sample = self._ask(self.small, question, temperature=SAMPLE_TEMPERATURE)
                first = Completion(
                    text,
                    first.tokens + sample.tokens,
                    first.seconds + sample.seconds,
                    first.tokens_in + sample.tokens_in,
                    first.tokens_out + sample.tokens_out,
                )
                disagree = final_answer(sample.text) != final_answer(text)
            # An empty reply (for example a thinking model that used its whole token budget) also escalates.
            if not text or text.upper().startswith(ESCALATE) or disagree:
                second = self._ask(self.large, question)
                if disagree:
                    why = "small model disagreed with itself"
                else:
                    why = "small model escalated" if text else "small model gave no answer"
                ans = Answer(
                    second.text, self.large, first.seconds + second.seconds, first.tokens, second.tokens, True, [why]
                )
            else:
                ans = Answer(first.text, self.small, first.seconds, tokens_small=first.tokens, reasons=reasons)
        if self.cache is not None and ans.text.strip():  # an empty cache is falsy (len 0)
            self.cache.put(question, ans.text)
        return ans
