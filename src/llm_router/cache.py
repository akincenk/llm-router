"""Semantic cache: answer a question that means the same as one already answered, without calling a model.

Guards that matter in production:
- Numbers must match exactly. "15% of 80" and "15% of 90" are almost identical in meaning but have different
  answers; embeddings alone would happily return the wrong one.
- Only final, non-empty answers are stored (never an ESCALATE reply or a failed call).
- Bounded size (oldest out first) and an optional time-to-live.
"""

import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text: str) -> tuple[str, ...]:
    return tuple(n.replace(",", ".") for n in _NUMBER.findall(text))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class SemanticCache:
    def __init__(
        self,
        embed: Callable[[str], list[float]],
        threshold: float = 0.9,
        max_items: int = 1000,
        ttl_seconds: float | None = None,
    ):
        self.embed, self.threshold, self.max_items, self.ttl = embed, threshold, max_items, ttl_seconds
        self._items: OrderedDict[str, tuple[list[float], tuple[str, ...], str, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, question: str) -> tuple[str, float] | None:
        """(answer, similarity) of the closest stored question if close enough and its numbers match."""
        vec, nums, now = self.embed(question), numbers(question), time.time()
        best = None
        with self._lock:
            for q, (v, n, answer, at) in list(self._items.items()):
                if self.ttl and now - at > self.ttl:
                    del self._items[q]
                    continue
                if n != nums:
                    continue
                sim = _cosine(vec, v)
                if sim >= self.threshold and (best is None or sim > best[1]):
                    best = (answer, sim)
        return best

    def put(self, question: str, answer: str) -> None:
        vec = self.embed(question)
        with self._lock:
            self._items[question] = (vec, numbers(question), answer, time.time())
            self._items.move_to_end(question)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)
