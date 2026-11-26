"""llm-router ask "question" [--strategy rules] — answer one question and show which model was used and why."""

import argparse
import os

from llm_router.client import Client
from llm_router.router import Router


def main() -> None:
    p = argparse.ArgumentParser(prog="llm-router")
    sub = p.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--strategy", default="rules", choices=["small", "large", "rules", "cascade"])
    ask.add_argument("--small", default="gemma4:12b")
    ask.add_argument("--large", default="qwen3.8:27b")
    ask.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"))
    a = p.parse_args()
    client = Client(a.base_url, os.getenv("OPENAI_API_KEY", "local"))
    ans = Router(client.chat, a.small, a.large, a.strategy).ask(a.question)
    print(ans.text)
    print(f"\n[{ans.model}, {ans.seconds:.1f}s, escalated={ans.escalated}, reasons={', '.join(ans.reasons) or '-'}]")


if __name__ == "__main__":
    main()
