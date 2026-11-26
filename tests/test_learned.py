import random

import pytest

from llm_router.learned import LogisticModel, fit


def test_fit_separates_a_learnable_signal_and_round_trips(tmp_path):
    rng = random.Random(1)
    xs = [[rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(5, 0.1)] for _ in range(200)]
    ys = [int(x[0] + 0.5 * x[1] > 0) for x in xs]
    m = fit(xs, ys, l2=0.1, epochs=400, lr=0.5)
    acc = sum((m.prob(x) > 0.5) == bool(y) for x, y in zip(xs, ys, strict=True)) / len(xs)
    assert acc > 0.9
    assert m.prob([2, 1, 5]) > 0.9 and m.prob([-2, -1, 5]) < 0.1
    m.save(tmp_path / "w.json")
    assert LogisticModel.load(tmp_path / "w.json").prob([2, 1, 5]) == m.prob([2, 1, 5])


def test_a_constant_feature_does_not_divide_by_zero():
    m = fit([[1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [4.0, 3.0]], [0, 0, 1, 1], epochs=200, lr=0.5)
    assert m.prob([4.0, 3.0]) > m.prob([1.0, 3.0])


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        fit([[1.0]], [2])
    with pytest.raises(ValueError):
        fit([], [])
