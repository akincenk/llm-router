import io
import json
import time
from unittest.mock import patch

import pytest

from llm_router.cache import SemanticCache, numbers
from llm_router.client import Client, Completion
from llm_router.metrics import summary
from llm_router.router import ESCALATE, Router, difficulty

VOCAB = ["capital", "canada", "city", "main", "percent", "of", "boxes", "pallet", "ottawa", "what", "is"]


def embed(text):
    words = text.lower().replace("?", "").replace("%", " percent ").split()
    return [words.count(w) + 0.01 for w in VOCAB]


class FakeChat:
    def __init__(self, small_reply="small answer", large_reply="large answer"):
        self.calls, self.small_reply, self.large_reply = [], small_reply, large_reply

    def __call__(self, model, messages):
        self.calls.append(model)
        text = self.small_reply if model == "small" else self.large_reply
        return Completion(text, 10 if model == "small" else 100, 0.01)


def test_difficulty_is_explainable():
    assert difficulty("What is the capital of Canada?") == (False, [])
    hard, why = difficulty("A warehouse has 120 boxes. Each pallet holds 18. After 3 pallets ship, how many remain?")
    assert hard and "3+ numbers" in why


def test_rules_send_easy_to_small_and_hard_to_large():
    chat = FakeChat()
    r = Router(chat, "small", "large", "rules")
    assert r.ask("What is the capital of Canada?").model == "small"
    assert r.ask("Explain step by step why each of the 3 pallets of 18 boxes from 120 remain?").model == "large"


def test_cascade_escalates_only_when_the_small_model_asks():
    r = Router(FakeChat(small_reply=ESCALATE), "small", "large", "cascade")
    a = r.ask("hard question")
    assert (a.model, a.escalated, a.tokens_small, a.tokens_large) == ("large", True, 10, 100)
    b = Router(FakeChat(), "small", "large", "cascade").ask("easy question")
    assert (b.model, b.escalated) == ("small", False)


@pytest.mark.parametrize("strategy", ["rules", "cascade"])
def test_an_empty_small_answer_goes_to_the_large_model(strategy):
    chat = FakeChat(small_reply="")
    a = Router(chat, "small", "large", strategy).ask("What is the capital of Canada?")
    assert (a.text, a.escalated, chat.calls) == ("large answer", True, ["small", "large"])
    assert a.reasons == ["small model gave no answer"]
    only_small = Router(FakeChat(small_reply=""), "small", "large", "small").ask("What is the capital of Canada?")
    assert only_small.model == "small"  # the baseline never falls back


def test_cache_answers_a_paraphrase_without_calling_a_model():
    chat = FakeChat()
    r = Router(chat, "small", "large", "rules", SemanticCache(embed, threshold=0.9))
    r.ask("What is the capital of Canada?")
    again = r.ask("what is the capital of canada")
    assert again.model == "cache" and chat.calls == ["small"]


def test_cache_never_mixes_up_numbers():
    cache = SemanticCache(embed, threshold=0.5)
    cache.put("What is 15% of 80?", "12")
    assert cache.get("What is 15% of 90?") is None
    assert cache.get("what is 15 % of 80") is not None
    assert numbers("1,5 and 2.5 and 7") == ("1.5", "2.5", "7")


def test_empty_answers_are_not_cached():
    r = Router(FakeChat(small_reply="  "), "small", "large", "small", SemanticCache(embed))
    r.ask("What is the capital of Canada?")
    assert len(r.cache) == 0


def test_cache_size_limit_and_ttl():
    cache = SemanticCache(embed, threshold=0.5, max_items=2, ttl_seconds=0.05)
    for q in ("capital", "pallet", "boxes"):
        cache.put(q, q)
    assert len(cache) == 2 and cache.get("capital") is None  # oldest evicted
    time.sleep(0.06)
    assert cache.get("boxes") is None  # expired


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError):
        Router(FakeChat(), "s", "l", "random")


def test_summary_counts_large_share_cache_hits_and_accuracy():
    chat = FakeChat()
    r = Router(chat, "small", "large", "large", SemanticCache(embed, threshold=0.9))
    answers = [r.ask("What is the capital of Canada?"), r.ask("what is the capital of canada")]
    s = summary(answers, [True, True])
    assert (s["large_model_share"], s["cache_hits"], s["accuracy"], s["tokens_large"]) == (0.5, 1, 1.0, 100)


def test_client_parses_chat_and_embeddings():
    chat = {"choices": [{"message": {"content": "Ottawa"}}], "usage": {"total_tokens": 5}}
    emb = {"data": [{"index": 1, "embedding": [0.2]}, {"index": 0, "embedding": [0.1]}]}
    with patch(
        "urllib.request.urlopen",
        side_effect=[io.BytesIO(json.dumps(chat).encode()), io.BytesIO(json.dumps(emb).encode())],
    ):
        c = Client("http://127.0.0.1:11434/v1")
        assert c.chat("m", []).text == "Ottawa"
        assert c.embed("e", ["a", "b"]) == [[0.1], [0.2]]
