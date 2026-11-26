"""Routing on a public benchmark: GSM8K grade-school maths word problems (OpenAI, MIT licence).

The question set is not written by me, so the router cannot be tuned to it by accident. Method (second run):

1. Train problems come from the GSM8K train file; test problems are GSM8K-Platinum (MadryLab, MIT): the GSM8K
   test set with mislabeled answers fixed and ambiguous questions removed (1,209 of 1,319 kept). Train and test
   never share a problem.
2. Ask every problem to the small model, the large model, the small model with the cascade note (it may say
   ESCALATE) and twice more to the small model sampled at temperature 0.8 (for consistency). The small model runs
   with reasoning_effort "none": with thinking on, Gemma 4 sometimes thinks until any token budget runs out and
   returns nothing. Every reply is stored, so a crashed run resumes where it stopped.
3. Embed every problem; train the learned router on the train problems (label: was the small model right?).
4. Simulate every strategy on the test problems from the stored replies, and pick every setting on train only.
   Two references bracket the result: a random router that sends the same share to the large model, and an
   oracle that knows which model is right.

The first run (2026-11-26, commit 67c5021, load_v1): 500 problems from the original test file, halves for train
and test, small model thinking with a 2,048-token cap. Its report stays in evals/results-gsm8k.json.

Usage:
  python -m evals.gsm8k --train 500 --cache evals/gsm8k-v2-replies.json --json evals/results-gsm8k-v2.json
"""

import argparse
import json
import math
import os
import random
import urllib.request
from collections import Counter
from pathlib import Path

from llm_router.client import Client
from llm_router.learned import fit
from llm_router.router import CASCADE_NOTE, ESCALATE, SAMPLE_TEMPERATURE, SYSTEM, difficulty, final_answer

BASE = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"
PLATINUM = (
    "https://datasets-server.huggingface.co/rows?dataset=madrylab%2Fgsm8k-platinum&config=main&split=test"
    "&offset={}&length=100"
)
DATA = Path(__file__).parent / "data"


def _rows(name: str, download) -> list[dict]:
    path = DATA / name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in download()))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def _platinum() -> list[dict]:
    rows, total = [], None
    while total is None or len(rows) < total:
        page = json.loads(_get(PLATINUM.format(len(rows))))
        total = page["num_rows_total"]
        if not page["rows"]:
            break
        rows += [r["row"] for r in page["rows"]]
    if len(rows) != total:
        raise RuntimeError(f"GSM8K-Platinum: got {len(rows)} of {total} rows")
    return rows


def _items(rows: list[dict]) -> list[dict]:
    return [{"q": r["question"], "answer": gold(r["answer"])} for r in rows]


def load_v1(n: int, seed: int = 0) -> list[dict]:
    """The first measurement (2026-11-26, commit 67c5021): n problems sampled from the original GSM8K test file."""
    rows = _rows("gsm8k-test.jsonl", lambda: [json.loads(x) for x in _get(BASE + "test.jsonl").splitlines() if x])
    return random.Random(seed).sample(_items(rows), n)


def load(n_train: int, n_test: int | None, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Train on problems from the GSM8K train file; test on GSM8K-Platinum (MIT, MadryLab 2025), the GSM8K test
    set with wrong labels fixed and ambiguous questions removed. n_test None means all 1,209."""
    train = _rows("gsm8k-train.jsonl", lambda: [json.loads(x) for x in _get(BASE + "train.jsonl").splitlines() if x])
    test = _items(_rows("gsm8k-platinum-test.jsonl", _platinum))
    rng = random.Random(seed)
    return rng.sample(_items(train), n_train), (test if n_test is None else rng.sample(test, n_test))


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
        "sample2": (small, "", SAMPLE_TEMPERATURE),
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
    if strategy == "fallback":  # control: only the empty-reply fallback, no routing decision at all
        return use_large(small) if not small["text"].strip() else use_small(small)
    if strategy == "cascade":
        text = rec["cascade"]["text"].strip()
        return use_large(rec["cascade"]) if not text or text.upper().startswith(ESCALATE) else use_small(rec["cascade"])
    if strategy == "consistency":
        if not small["text"].strip():
            return use_large(small)
        if final_answer(rec["sample"]["text"]) != final_answer(small["text"]):
            return use_large(small, rec["sample"])
        return use_small(small, rec["sample"])
    if strategy in ("consistency3", "majority3"):
        texts = [small, rec["sample"], rec["sample2"]]
        if not small["text"].strip():
            return use_large(small)
        votes = Counter(final_answer(r["text"]) for r in texts if r["text"].strip())
        top, n = votes.most_common(1)[0]
        if n == 3 or (strategy == "majority3" and n == 2):
            return {"ok": top == item["answer"], "usd": sum(cost(r, prices["small"]) for r in texts), "large": False}
        return use_large(*texts)
    if strategy == "learned_consistency":  # the learned router first; questions it calls easy get the consistency check
        if model.prob(rec["embedding"]) < threshold:
            return use_large()
        return simulate("consistency", item, rec, prices)
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


def pick_threshold(model, train: list[dict], store: dict, prices: dict, strategy: str = "learned") -> float:
    target = score("large", train, store, prices)["accuracy"] - 0.01
    best = 1.0  # threshold 1.0 sends everything to the large model
    for t in [i / 100 for i in range(1, 100)]:
        s = score(strategy, train, store, prices, model=model, threshold=t)
        if s["accuracy"] >= target and t < best:
            best = t
    return best


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """Chance that a random positive scores higher than a random negative (ties count half)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def signals(items: list[dict], store: dict, model) -> dict:
    """How well each escalation signal predicts two targets (arXiv 2610.07354 checklist): "small_wrong" and
    "escalation_helps" (small wrong and large right). Question-only signals are the control: a live signal
    that does not beat them only tracks how hard a question looks."""
    small_ok = [correct(store[it["q"]]["small"]["text"], it) for it in items]
    large_ok = [correct(store[it["q"]]["large"]["text"], it) for it in items]

    def agree(rec):
        texts = [rec[k]["text"] for k in ("small", "sample", "sample2") if k in rec and rec[k]["text"].strip()]
        return Counter(map(final_answer, texts)).most_common(1)[0][1] if texts else 0

    def cascade_flag(rec):
        text = rec["cascade"]["text"].strip()
        return float(not text or text.upper().startswith(ESCALATE))

    sig = {
        "question_only/rules_signal_count": [len(difficulty(it["q"])[1]) for it in items],
        "question_only/learned_1_minus_p": [1 - model.prob(store[it["q"]]["embedding"]) for it in items],
        "live/disagree_2_samples": [
            float(final_answer(store[it["q"]]["small"]["text"]) != final_answer(store[it["q"]]["sample"]["text"]))
            for it in items
        ],
        "live/cascade_escalate": [cascade_flag(store[it["q"]]) for it in items],
    }
    if all("sample2" in store[it["q"]] for it in items):
        sig["live/disagreement_of_3"] = [3 - agree(store[it["q"]]) for it in items]
    targets = {
        "small_wrong": [not s for s in small_ok],
        "escalation_helps": [(not s) and b for s, b in zip(small_ok, large_ok, strict=True)],
    }
    return {
        "auroc": {name: {t: auroc(v, y) for t, y in targets.items()} for name, v in sig.items()},
        "headroom_oracle_minus_large": round(
            (sum(s or b for s, b in zip(small_ok, large_ok, strict=True)) - sum(large_ok)) / len(items), 4
        ),
        "positives": {t: sum(y) for t, y in targets.items()},
    }


def pick_l2(xs: list, ys: list[int], folds: int = 5, grid=(1.0, 10.0, 100.0, 1000.0, 10000.0)) -> float:
    """L2 penalty with the lowest cross-validated log-loss on the training set (calmer probabilities)."""
    best, best_loss = grid[0], math.inf
    for l2 in grid:
        loss = 0.0
        for f in range(folds):
            tr = [i for i in range(len(xs)) if i % folds != f]
            m = fit([xs[i] for i in tr], [ys[i] for i in tr], l2=l2)
            for i in range(f, len(xs), folds):
                p = min(max(m.prob(xs[i]), 1e-6), 1 - 1e-6)
                loss -= math.log(p if ys[i] else 1 - p)
        if loss < best_loss:
            best, best_loss = l2, loss
    return best


class OutOfFold:
    """Out-of-fold probabilities for the training problems, so the threshold is not picked on predictions the
    model made for its own training examples (those are overconfident: the first run picked 0.01)."""

    def __init__(self, xs: list, ys: list[int], l2: float, folds: int = 5):
        self.p = {}
        for f in range(folds):
            tr = [i for i in range(len(xs)) if i % folds != f]
            m = fit([xs[i] for i in tr], [ys[i] for i in tr], l2=l2)
            for i in range(f, len(xs), folds):
                self.p[id(xs[i])] = m.prob(xs[i])

    def prob(self, x: list[float]) -> float:
        return self.p[id(x)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"))
    p.add_argument("--small", default="gemma4:12b")
    p.add_argument("--large", default="qwen3.8:27b")
    p.add_argument("--embed", default="embeddinggemma2-bf16")
    p.add_argument("--train", type=int, default=500, help="problems from the GSM8K train file")
    p.add_argument("--test", type=int, help="problems from GSM8K-Platinum (default: all 1,209)")
    p.add_argument("--small-reasoning", default="none", help="reasoning_effort for the small model ('' = default)")
    p.add_argument("--cache", default="evals/gsm8k-v2-replies.json")
    p.add_argument("--price-small", default="0.10,0.50", help="dollars per million tokens: input,output")
    p.add_argument("--price-large", default="4,20", help="dollars per million tokens: input,output")
    p.add_argument("--weights", help="also save the trained router here")
    p.add_argument("--json", help="write the report here")
    a = p.parse_args()
    prices = {
        "small": tuple(map(float, a.price_small.split(","))),
        "large": tuple(map(float, a.price_large.split(","))),
    }
    train, test = load(a.train, a.test)
    options = {a.small: {"reasoning_effort": a.small_reasoning}} if a.small_reasoning else {}
    client = Client(a.base_url, os.getenv("OPENAI_API_KEY", "local"), options=options)
    store = collect(client, train + test, a.small, a.large, a.embed, Path(a.cache))
    ys = [int(correct(store[it["q"]]["small"]["text"], it)) for it in train]
    xs = [store[it["q"]]["embedding"] for it in train]
    l2 = pick_l2(xs, ys)
    model = fit(xs, ys, l2=l2)
    if a.weights:
        model.save(a.weights)
    oof = OutOfFold(xs, ys, l2)
    threshold = pick_threshold(oof, train, store, prices)
    report = {
        "models": {"small": a.small, "large": a.large, "embed": a.embed, "options": options},
        "data": "train: GSM8K train file; test: GSM8K-Platinum",
        "n_train": len(train),
        "n_test": len(test),
        "threshold": threshold,
        "l2": l2,
    }
    for s in ("small", "large", "fallback", "rules", "cascade", "consistency", "consistency3", "majority3"):
        report[s] = score(s, test, store, prices)
    t_lc = pick_threshold(oof, train, store, prices, "learned_consistency")
    report["learned_consistency"] = score("learned_consistency", test, store, prices, model=model, threshold=t_lc)
    report["learned_consistency"]["threshold"] = t_lc
    report["learned"] = score("learned", test, store, prices, model=model, threshold=threshold)
    report["learned_sweep"] = {
        str(t): score("learned", test, store, prices, model=model, threshold=t) for t in (0.5, 0.6, 0.7, 0.8, 0.9)
    }
    report.update(references(test, store, prices, report["learned"]["large_share"]))
    report["signals"] = signals(test, store, model)
    print(json.dumps(report, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
