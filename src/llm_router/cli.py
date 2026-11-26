"""llm-router ask "question" [--strategy rules] — answer one question and show which model was used and why."""

import argparse
import os

from llm_router.client import Client
from llm_router.learned import LogisticModel
from llm_router.router import STRATEGIES, Router


def main() -> None:
    p = argparse.ArgumentParser(prog="llm-router")
    sub = p.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--strategy", default="rules", choices=STRATEGIES)
    ask.add_argument("--small", default="gemma4:12b")
    ask.add_argument("--large", default="qwen3.8:27b")
    ask.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"))
    ask.add_argument("--weights", help="learned strategy: a model saved by evals.gsm8k (JSON)")
    ask.add_argument("--embed", default="embeddinggemma2-bf16", help="learned strategy: embedding model")
    ask.add_argument("--threshold", type=float, default=0.5, help="learned strategy: route to large below this")
    a = p.parse_args()
    client = Client(a.base_url, os.getenv("OPENAI_API_KEY", "local"))
    predictor = None
    if a.strategy == "learned":
        if not a.weights:
            p.error("--strategy learned needs --weights")
        model = LogisticModel.load(a.weights)
        predictor = lambda q: model.prob(client.embed(a.embed, [q])[0])  # noqa: E731
    ans = Router(client.chat, a.small, a.large, a.strategy, predictor=predictor, threshold=a.threshold).ask(a.question)
    print(ans.text)
    print(f"\n[{ans.model}, {ans.seconds:.1f}s, escalated={ans.escalated}, reasons={', '.join(ans.reasons) or '-'}]")


if __name__ == "__main__":
    main()
