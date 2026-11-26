"""Summaries for a stream of answers: how much went to the large model, the cache, and how long it took."""

from llm_router.router import Answer


def summary(answers: list[Answer], correct: list[bool] | None = None) -> dict:
    n = len(answers) or 1
    large = [a for a in answers if a.tokens_large]
    out = {
        "requests": len(answers),
        "large_model_share": round(len(large) / n, 3),
        "cache_hits": sum(a.model == "cache" for a in answers),
        "escalations": sum(a.escalated for a in answers),
        "tokens_small": sum(a.tokens_small for a in answers),
        "tokens_large": sum(a.tokens_large for a in answers),
        "avg_seconds": round(sum(a.seconds for a in answers) / n, 2),
    }
    if correct is not None:
        out["accuracy"] = round(sum(correct) / (len(correct) or 1), 3)
    return out
