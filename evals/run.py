"""Compare routing strategies on questions with checkable answers.

1. Each strategy (small, large, rules, cascade) answers the 40 base questions (20 easy, 20 hard) without a cache.
2. The best-value strategy then runs behind the semantic cache over the base questions, 15 paraphrases
   (should be cache hits) and 5 number traps (near duplicates with different numbers: must NOT be cache hits).

Grading: the last non-empty line of the reply must contain every expected value (commas removed,
case-insensitive), and one value from each 'any' group.

Cost: every model call's prompt and completion tokens are priced per million tokens. By default the small model
is priced like Claude Haiku 5.5 ($0.10 in / $0.50 out) and the large one like Claude Opus 5.5 ($4 in / $20 out),
the list prices on platform.claude.com/docs/en/about-claude/pricing on 2026-11-25. The models that actually ran
are local, so the dollars show what the same token counts would cost at those prices.

Usage (fully local with Ollama):
  python -m evals.run --small gemma4:12b --large qwen3.8:27b --embed embeddinggemma2-bf16
"""

import argparse
import json
import os
from pathlib import Path

from llm_router.cache import SemanticCache
from llm_router.client import Client, Completion
from llm_router.metrics import summary
from llm_router.router import Answer, Router

QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text())


def graded(reply: str, item: dict) -> bool:
    lines = [line for line in reply.splitlines() if line.strip()]
    last = (lines[-1] if lines else "").replace(",", "").lower()
    return all(e.lower() in last for e in item["expect"]) and all(
        any(alt.lower() in last for alt in group) for group in item.get("any", [])
    )


class PricedChat:
    """Wraps chat() and adds up the dollar cost of every call at the given prices."""

    def __init__(self, chat, prices: dict[str, tuple[float, float]]):
        self.chat, self.prices, self.usd = chat, prices, 0.0

    def __call__(self, model: str, messages: list[dict]) -> Completion:
        c = self.chat(model, messages)
        p_in, p_out = self.prices[model]
        self.usd += (c.tokens_in * p_in + c.tokens_out * p_out) / 1e6
        return c


def run_set(router: Router, items: list[dict]) -> tuple[dict, list[dict]]:
    answers, ok, rows = [], [], []
    for item in items:
        try:
            a = router.ask(item["q"])
        except OSError as e:  # timeout or connection error: counts as a wrong answer, the run goes on
            a = Answer(f"error: {e}", "error")
        answers.append(a)
        ok.append(graded(a.text, item))
        rows.append(
            {"q": item["q"], "ok": ok[-1], "model": a.model, "escalated": a.escalated, "s": round(a.seconds, 1)}
        )
        print(json.dumps(rows[-1]), flush=True)
    out = summary(answers, ok)
    if isinstance(router.chat, PricedChat):
        out["cost_usd"] = round(router.chat.usd, 4)
    return out, rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"))
    p.add_argument("--small", default="gemma4:12b")
    p.add_argument("--large", default="qwen3.8:27b")
    p.add_argument("--embed", default="embeddinggemma2-bf16")
    p.add_argument("--cache-strategy", default="cascade")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--price-small", default="0.10,0.50", help="dollars per million tokens: input,output")
    p.add_argument("--price-large", default="4,20", help="dollars per million tokens: input,output")
    p.add_argument("--json", help="also write the raw results here")
    a = p.parse_args()
    client = Client(a.base_url, os.getenv("OPENAI_API_KEY", "local"))
    base = QUESTIONS["easy"] + QUESTIONS["hard"]
    report, raw = {"small": a.small, "large": a.large}, {}
    prices = {
        a.small: tuple(map(float, a.price_small.split(","))),
        a.large: tuple(map(float, a.price_large.split(","))),
    }

    def priced():
        return PricedChat(client.chat, prices)

    for strategy in ("small", "large", "rules", "cascade"):
        print(f"# {strategy}", flush=True)
        report[strategy], raw[strategy] = run_set(Router(priced(), a.small, a.large, strategy), base)

    print(f"# {a.cache_strategy} + cache", flush=True)
    cache = SemanticCache(lambda t: client.embed(a.embed, [t])[0], threshold=a.threshold)
    router = Router(priced(), a.small, a.large, a.cache_strategy, cache)
    cached, rows = run_set(router, base + QUESTIONS["paraphrases"] + QUESTIONS["number_traps"])
    para = rows[len(base) : len(base) + len(QUESTIONS["paraphrases"])]
    traps = rows[-len(QUESTIONS["number_traps"]) :]
    cached["paraphrase_hits"] = f"{sum(r['model'] == 'cache' for r in para)}/{len(para)}"
    cached["paraphrase_correct"] = f"{sum(r['ok'] for r in para)}/{len(para)}"
    cached["trap_false_hits"] = f"{sum(r['model'] == 'cache' for r in traps)}/{len(traps)}"
    cached["trap_correct"] = f"{sum(r['ok'] for r in traps)}/{len(traps)}"
    report["with_cache"], raw["with_cache"] = cached, rows

    print(json.dumps(report, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps({"report": report, "rows": raw}, indent=2))


if __name__ == "__main__":
    main()
