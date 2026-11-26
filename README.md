# llm-router

[![CI](https://github.com/akincenk/llm-router/actions/workflows/ci.yml/badge.svg)](https://github.com/akincenk/llm-router/actions/workflows/ci.yml)

Send each question to the cheapest model that can answer it, and skip the model entirely when the same question was already answered. Standard library only; works with any OpenAI-compatible endpoint, including fully local models through [Ollama](https://ollama.com).

On a 40-question test set, routing by explainable rules kept the accuracy of always using the large model (**100%**) at **58% lower cost**. A semantic cache in front answered **14 of 15** reworded questions without calling any model and was **never fooled by changed numbers (0 of 5)**.

## How it works

| Part | What it does |
| --- | --- |
| `rules` strategy | An explainable difficulty score (length, 3+ numbers, reasoning words such as "each", "after", "probability", several questions, code). Two or more signals send the question to the large model; every answer records why. |
| `cascade` strategy | The small model answers first and may reply `ESCALATE` when unsure; then the large model answers. |
| Fallback | In `rules` and `cascade`, an empty reply from the small model (for example a thinking model that used its whole token budget) also goes to the large model. |
| `SemanticCache` | Embeds the question and returns a stored answer when cosine similarity is at least 0.9. **Numbers must match exactly**, so "15% of 80" never returns the answer to "15% of 90". Bounded size, optional time-to-live, thread-safe. |
| Metrics | Per run: large-model share, cache hits, escalations, tokens per model, average seconds and accuracy; the evaluation adds the dollar cost of input and output tokens. |
| Baselines | `small` and `large` always use one model, so every strategy is compared against them. |

## Use it

```bash
pip install -e .
ollama pull gemma4:12b && ollama pull qwen3.8:27b
llm-router ask "A warehouse has 1240 boxes and each pallet holds 36. How many full pallets?" --strategy rules
```

```python
from llm_router.cache import SemanticCache
from llm_router.client import Client
from llm_router.router import Router

client = Client("http://127.0.0.1:11434/v1")
cache = SemanticCache(lambda text: client.embed("embeddinggemma2-bf16", [text])[0])
router = Router(client.chat, small="gemma4:12b", large="qwen3.8:27b", strategy="rules", cache=cache)
answer = router.ask("What is 15% of 80?")
print(answer.text, answer.model, answer.reasons)
```

## Evaluation

`python -m evals.run` asks 40 questions with checkable answers (20 easy, 20 multi-step word problems) through every strategy, then runs the cascade behind the cache on the same 40 plus 15 reworded questions and 5 number traps (the same wording as an earlier question with different numbers). A reply counts as correct when its last line contains the expected value.

Models: Gemma 4 12B (small) and Qwen 3.8 27B (large), both local through Ollama on a Mac Studio, EmbeddingGemma 2 for the cache, 2026-11-25. The models ran for free on my machine, so cost is the **measured input and output tokens priced at Anthropic list prices** (small priced like Claude Haiku 5.5: $0.10 / $0.50 per million input / output tokens; large like Claude Opus 5.5: $4 / $20). `--price-small` and `--price-large` take any other prices. Every answer of this run is in [`evals/results-2026-11-25.json`](evals/results-2026-11-25.json).

| Strategy (40 questions) | Accuracy | Sent to large model | Cost | Per million questions | Avg seconds |
| --- | --- | --- | --- | --- | --- |
| Always large | 100% | 100% | $0.124 | $3,105 | 2.2 |
| Always small | 97.5% | 0% | $0.008 | $200 | 3.1 |
| **Rules** | **100%** | **22.5%** | **$0.052** | **$1,298 (−58%)** | 2.9 |
| Cascade | 97.5% | 7.5% | $0.026 | $658 (−79%) | 4.9 |

| Cascade + semantic cache (60 questions) | Result |
| --- | --- |
| Reworded questions answered from the cache | 14 / 15 (all 15 correct) |
| Number traps wrongly answered from the cache | **0 / 5** (all 5 correct) |
| Accuracy over all 60 | 98.3% |
| Cost | $0.033, i.e. $543 per million questions (−82.5% against always large) |

What the numbers mean, honestly:

- **Rules is the safe choice**: the same accuracy as the large model for less than half the price. **Cascade is cheaper but lost one hard question**: the small model rarely knows when it is wrong, so it escalated only 3 times.
- The large model was not slower here. Qwen 3.8 gives short answers while Gemma 4 writes out its working, so routing saved money, not time. With hosted models, where the large model is usually slower, the time saving would come on top.
- The first cascade run exposed a real bug: Gemma 4 sometimes thinks through its whole token budget and returns nothing. Those empty replies counted as answers. They now go to the large model, which raised cascade accuracy from 85% to 97.5%.
- The questions are easy enough that both models score 97.5% or better, so this shows the cost saving at equal quality, not which model is smarter. 40 questions are a smoke test; run it on your own traffic before trusting the percentages.
- A cache repeats whatever answer it stored, including a wrong one, and costs one embedding call per question (not counted above).

## Tests

`pytest` runs 15 tests with a fake model and a fake embedding, so routing, escalation, the empty-reply fallback, cache hits, the number guard, size limit and time-to-live, metrics, cost and grading are checked without a network or a GPU.

```
src/llm_router/
  client.py   OpenAI-compatible chat and embeddings (stdlib), with token counts
  router.py   difficulty rules, strategies, escalation and fallback
  cache.py    semantic cache with the number guard, size limit and TTL
  metrics.py  per-run summary
  cli.py      llm-router ask
evals/        40 questions, 15 paraphrases, 5 number traps and the runner
```

## Why this exists

My own AI assistant routes work between a local model and paid cloud models with spending caps. This repository is a clean, public version of that idea with the measurements in the open.

## License

MIT
