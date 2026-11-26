"""Routing on a public benchmark: GSM8K grade-school maths word problems (OpenAI, MIT licence).

The question set is not written by me, so the router cannot be tuned to it by accident. Method:

1. Sample N problems from the GSM8K test file with a fixed seed; the first half trains, the second half tests.
2. Ask every problem once to the small model, once to the large model, once to the small model with the cascade
   note (it may say ESCALATE) and once more to the small model sampled at temperature 0.8 (for consistency).
   Every reply is stored, so a crashed run resumes where it stopped.
3. Embed every problem; train the learned router on the training half (label: was the small model right?).
4. Simulate every strategy on the test half from the stored replies. A decision that depends only on the
   question (rules, learned) needs no new model call, so the learned router can be swept over thresholds.
   The threshold reported is picked on the training half: the cheapest one whose training accuracy stays within
   one point of the large model's. Two references bracket the result: a random router that sends the same share
   to the large model, and an oracle that knows which model is right.

Usage:
  python -m evals.gsm8k --n 500 --cache evals/gsm8k-replies.json --json evals/results-gsm8k.json
"""

import argparse
import json
import os
import random
import urllib.request
from pathlib import Path

from llm_router.client import Client
from llm_router.learned import fit
from llm_router.router import CASCADE_NOTE, ESCALATE, SAMPLE_TEMPERATURE, SYSTEM, difficulty, final_answer

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
DATA = Path(__file__).parent / "data" / "gsm8k-test.jsonl"


def load(n: int, seed: int = 0) -> list[dict]:
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            DATA.write_bytes(r.read())
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    items = [{"q": r["question"], "answer": gold(r["answer"])} for r in rows]
    return random.Random(seed).sample(items, n)


def gold(solution: str) -> str:
    """GSM8K puts the final answer after '####'."""
    return final_answer(solution.split("####")[-1].strip())


def correct(reply: str, item: dict) -> bool:
    return bool(reply.strip()) and final_answer(reply) == item["answer"]


def collect(client: Client, items: list[dict], small: str, large: str, embed: str, cache: Path) -> dict:
    store = json.loads(cache.read_text()) if cache.exists() else {}
    calls = {
        "small": (small, "", 0),
        "large": (large, "", 0),
        "cascade": (small, CASCADE_NOTE, 0),
        "sample": (small, "", SAMPLE_TEMPERATURE),
    }
    for i, item in enumerate(items):
        rec = store.setdefault(item["q"], {})
        for key, (model, note, temp) in calls.items():
            if key in rec:
                continue
            msgs = [{"role": "system", "content": SYSTEM + note}, {"role": "user", "content": item["q"]}]
            try:
                c = client.chat(model, msgs, temperature=temp)
                rec[key] = {"text": c.text, "in": c.tokens_in, "out": c.tokens_out, "s": round(c.seconds, 2)}
            except OSError as e:  # recorded as an empty reply: a wrong answer, and the run goes on
                rec[key] = {"text": "", "in": 0, "out": 0, "s": 0, "error": str(e)[:200]}
        if "embedding" not in rec:
            rec["embedding"] = client.embed(embed, [item["q"]])[0]
        if i % 10 == 9 or i == len(items) - 1:
            cache.write_text(json.dumps(store))
            print(f"{i + 1}/{len(items)} collected", flush=True)
    return store


def cost(rec: dict, prices: dict[str, tuple[float, float]]) -> float:
    p_in, p_out = prices
    return (rec["in"] * p_in + rec["out"] * p_out) / 1e6


def simulate(strategy: str, item: dict, rec: dict, prices: dict, model=None, threshold: float = 0.5) -> dict:
    """What the Router would have done with these stored replies: correct?, dollars, went to large?"""
    small, large = rec["small"], rec["large"]

    def use_large(*paid):
        return {
            "ok": correct(large["text"], item),
            "usd": sum(cost(r, prices["small"]) for r in paid) + cost(large, prices["large"]),
            "large": True,
        }

    def use_small(*paid):
        return {
            "ok": correct(paid[0]["text"], item),
            "usd": sum(cost(r, prices["small"]) for r in paid),
            "large": False,
        }

    if strategy == "large":
        return use_large()
    if strategy == "small":
        return use_small(small)
    if strategy == "cascade":
        text = rec["cascade"]["text"].strip()
        return use_large(rec["cascade"]) if not text or text.upper().startswith(ESCALATE) else use_small(rec["cascade"])
    if strategy == "consistency":
        if not small["text"].strip():
            return use_large(small)
        if final_answer(rec["sample"]["text"]) != final_answer(small["text"]):
            return use_large(small, rec["sample"])
        return use_small(small, rec["sample"])
    hard = difficulty(item["q"])[0] if strategy == "rules" else model.prob(rec["embedding"]) < threshold
    if hard:
        return use_large()
    return use_large(small) if not small["text"].strip() else use_small(small)


def score(strategy: str, items: list[dict], store: dict, prices: dict, **kw) -> dict:
    rows = [simulate(strategy, it, store[it["q"]], prices, **kw) for it in items]
    n = len(rows)
    return {
        "accuracy": round(sum(r["ok"] for r in rows) / n, 4),
        "large_share": round(sum(r["large"] for r in rows) / n, 4),
        "usd_per_million": round(sum(r["usd"] for r in rows) / n * 1e6, 2),
    }


def references(items: list[dict], store: dict, prices: dict, share: float) -> dict:
    small_ok = [correct(store[it["q"]]["small"]["text"], it) for it in items]
    large_ok = [correct(store[it["q"]]["large"]["text"], it) for it in items]
    n = len(items)
    c_small = sum(cost(store[it["q"]]["small"], prices["small"]) for it in items) / n
    c_large = sum(cost(store[it["q"]]["large"], prices["large"]) for it in items) / n
    oracle = [s or b for s, b in zip(small_ok, large_ok, strict=True)]
    oracle_cost = (
        sum(
            cost(store[it["q"]]["small"], prices["small"]) if ok else cost(store[it["q"]]["large"], prices["large"])
            for it, ok in zip(items, small_ok, strict=True)
        )
        / n
    )
    return {
        "random_same_share": {
            "accuracy": round(share * sum(large_ok) / n + (1 - share) * sum(small_ok) / n, 4),
            "large_share": share,
            "usd_per_million": round((share * c_large + (1 - share) * c_small) * 1e6, 2),
        },
        "oracle": {
            "accuracy": round(sum(oracle) / n, 4),
            "large_share": round(sum(not s for s in small_ok) / n, 4),
            "usd_per_million": round(oracle_cost * 1e6, 2),
        },
    }


def pick_threshold(model, train: list[dict], store: dict, prices: dict) -> float:
    target = score("large", train, store, prices)["accuracy"] - 0.01
    best = 1.0  # threshold 1.0 sends everything to the large model
    for t in [i / 100 for i in range(1, 100)]:
        s = score("learned", train, store, prices, model=model, threshold=t)
        if s["accuracy"] >= target and t < best:
            best = t
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"))
    p.add_argument("--small", default="gemma4:12b")
    p.add_argument("--large", default="qwen3.8:27b")
    p.add_argument("--embed", default="embeddinggemma2-bf16")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--cache", default="evals/gsm8k-replies.json")
    p.add_argument("--price-small", default="0.10,0.50", help="dollars per million tokens: input,output")
    p.add_argument("--price-large", default="4,20", help="dollars per million tokens: input,output")
    p.add_argument("--weights", help="also save the trained router here")
    p.add_argument("--json", help="write the report here")
    a = p.parse_args()
    prices = {
        "small": tuple(map(float, a.price_small.split(","))),
        "large": tuple(map(float, a.price_large.split(","))),
    }
    items = load(a.n)
    train, test = items[: a.n // 2], items[a.n // 2 :]
    client = Client(a.base_url, os.getenv("OPENAI_API_KEY", "local"))
    store = collect(client, items, a.small, a.large, a.embed, Path(a.cache))
    ys = [int(correct(store[it["q"]]["small"]["text"], it)) for it in train]
    model = fit([store[it["q"]]["embedding"] for it in train], ys)
    if a.weights:
        model.save(a.weights)
    threshold = pick_threshold(model, train, store, prices)
    report = {
        "models": {"small": a.small, "large": a.large, "embed": a.embed},
        "n_train": len(train),
        "n_test": len(test),
        "threshold": threshold,
    }
    for s in ("small", "large", "rules", "cascade", "consistency"):
        report[s] = score(s, test, store, prices)
    report["learned"] = score("learned", test, store, prices, model=model, threshold=threshold)
    report["learned_sweep"] = {
        str(t): score("learned", test, store, prices, model=model, threshold=t) for t in (0.5, 0.6, 0.7, 0.8, 0.9)
    }
    report.update(references(test, store, prices, report["learned"]["large_share"]))
    print(json.dumps(report, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
