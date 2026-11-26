"""A learned router: predict from a question's embedding whether the small model will answer it correctly.

Plain logistic regression, standard library only. Features are standardised with the training set's mean and
spread; L2 regularisation keeps the weights small on few examples. The router sends a question to the large
model when the predicted chance that the small model is right falls below a threshold, so one number trades
cost for accuracy.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


@dataclass
class LogisticModel:
    weights: list[float]
    bias: float
    mean: list[float]
    scale: list[float]

    def prob(self, x: list[float]) -> float:
        """Probability that the small model answers this question correctly."""
        z = self.bias + sum(w * (v - m) / s for w, v, m, s in zip(self.weights, x, self.mean, self.scale, strict=True))
        return _sigmoid(z)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__))

    @classmethod
    def load(cls, path: str | Path) -> "LogisticModel":
        return cls(**json.loads(Path(path).read_text()))


def fit(
    xs: list[list[float]],
    ys: list[int],
    l2: float = 1.0,
    epochs: int = 300,
    lr: float = 0.1,
    seed: int = 0,
) -> LogisticModel:
    """Full-batch gradient descent on the regularised log-loss. ys are 1 (small model right) or 0."""
    if not xs or len(xs) != len(ys) or set(ys) - {0, 1}:
        raise ValueError("need equal-length features and 0/1 labels")
    n, d = len(xs), len(xs[0])
    mean = [sum(x[j] for x in xs) / n for j in range(d)]
    scale = [math.sqrt(sum((x[j] - mean[j]) ** 2 for x in xs) / n) or 1.0 for j in range(d)]
    zs = [[(x[j] - mean[j]) / scale[j] for j in range(d)] for x in xs]
    rng = random.Random(seed)
    w = [rng.gauss(0, 0.01) for _ in range(d)]
    pos = sum(ys) / n
    b = math.log(pos / (1 - pos)) if 0 < pos < 1 else 0.0  # start at the base rate
    for _ in range(epochs):
        gw, gb = [l2 * wj / n for wj in w], 0.0
        for z, y in zip(zs, ys, strict=True):
            err = _sigmoid(b + sum(wj * zj for wj, zj in zip(w, z, strict=True))) - y
            gb += err / n
            for j in range(d):
                gw[j] += err * z[j] / n
        w = [wj - lr * g for wj, g in zip(w, gw, strict=True)]
        b -= lr * gb
    return LogisticModel(w, b, mean, scale)
