from evals.run import QUESTIONS, graded


def test_grading_uses_last_line_and_alternatives():
    item = {"expect": ["34", "16"]}
    assert graded("Work: 1240 / 36\n34 pallets, 16 left", item)
    assert not graded("34 pallets, 16 left\nDone.", item)
    assert graded("Total\n$1,380", {"expect": ["1380"]})
    assert graded("It has six sides", {"expect": [], "any": [["6", "six"]]})


def test_question_set_shape():
    assert len(QUESTIONS["easy"]) == len(QUESTIONS["hard"]) == 20
    assert len(QUESTIONS["paraphrases"]) == 15 and len(QUESTIONS["number_traps"]) == 5


def test_priced_chat_adds_dollars_per_model():
    from evals.run import PricedChat
    from llm_router.client import Completion

    chat = PricedChat(lambda m, msgs: Completion("x", 0, 0.0, 1_000_000, 100_000), {"s": (0.1, 0.5), "l": (4, 20)})
    chat("s", [])
    chat("l", [])
    assert round(chat.usd, 6) == 0.1 + 0.05 + 4 + 2
