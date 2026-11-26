from evals.gsm8k import correct, gold, references, score, simulate
from llm_router.learned import LogisticModel

PRICES = {"small": (1.0, 1.0), "large": (10.0, 10.0)}


def r(text, tokens=1_000_000):
    return {"text": text, "in": tokens, "out": 0, "s": 0}


def rec(small, large="9", cascade=None, sample=None, emb=(0.0,)):
    return {"small": r(small), "large": r(large), "cascade": r(cascade if cascade is not None else small),
            "sample": r(sample if sample is not None else small), "embedding": list(emb)}  # fmt: skip


ITEM = {"q": "What is 4 + 5?", "answer": "9"}


def test_gold_and_grading():
    assert gold("4 + 5 = <<4+5=9>>9\n#### 9") == "9"
    assert gold("... #### 1,200") == "1200"
    assert correct("so the total is\n9", ITEM) and not correct("", ITEM) and not correct("8", ITEM)


def test_simulated_strategies_pay_for_every_call_they_make():
    assert simulate("small", ITEM, rec("8"), PRICES) == {"ok": False, "usd": 1.0, "large": False}
    assert simulate("large", ITEM, rec("8"), PRICES) == {"ok": True, "usd": 10.0, "large": True}
    assert simulate("cascade", ITEM, rec("8", cascade="ESCALATE"), PRICES) == {"ok": True, "usd": 11.0, "large": True}
    assert simulate("consistency", ITEM, rec("9", sample="9"), PRICES) == {"ok": True, "usd": 2.0, "large": False}
    assert simulate("consistency", ITEM, rec("8", sample="7"), PRICES) == {"ok": True, "usd": 12.0, "large": True}
    assert simulate("rules", ITEM, rec(""), PRICES) == {"ok": True, "usd": 11.0, "large": True}  # empty reply escalates


def test_learned_threshold_decides():
    model = LogisticModel([0.0], 0.0, [0.0], [1.0])  # always 0.5
    assert simulate("learned", ITEM, rec("8"), PRICES, model=model, threshold=0.6)["large"]
    assert not simulate("learned", ITEM, rec("8"), PRICES, model=model, threshold=0.4)["large"]


def test_score_and_references():
    items = [ITEM, {"q": "What is 1 + 1?", "answer": "2"}]
    store = {ITEM["q"]: rec("9"), items[1]["q"]: rec("3", large="2")}
    assert score("small", items, store, PRICES) == {"accuracy": 0.5, "large_share": 0.0, "usd_per_million": 1e6}
    ref = references(items, store, PRICES, share=0.5)
    assert ref["oracle"] == {"accuracy": 1.0, "large_share": 0.5, "usd_per_million": 5.5e6}
    assert ref["random_same_share"]["accuracy"] == 0.75
