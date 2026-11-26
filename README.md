# llm-router

[![CI](https://github.com/akincenk/llm-router/actions/workflows/ci.yml/badge.svg)](https://github.com/akincenk/llm-router/actions/workflows/ci.yml)

Send each question to the cheapest model that can answer it, and skip the model entirely when the same question was already answered. Standard library only; works with any OpenAI-compatible endpoint, including fully local models through [Ollama](https://ollama.com).

On a 40-question test set, routing by explainable rules kept the accuracy of always using the large model (**100%**) at **58% lower cost**. A semantic cache in front answered **14 of 15** reworded questions without calling any model and was **never fooled by changed numbers (0 of 5)**.

On 250 held-out problems of the public **GSM8K** maths benchmark, **self-consistency routing matched the large model's accuracy (96.4%) at 51% lower cost**, and a **learned router** (logistic regression on question embeddings) reached 95.6% at 58% lower cost, against 80.1% for a random router sending the same share to the large model.

## How it works

| Part | What it does |
| --- | --- |
| `rules` strategy | An explainable difficulty score (length, 3+ numbers, reasoning words such as "each", "after", "probability", several questions, code). Two or more signals send the question to the large model; every answer records why. |
| `cascade` strategy | The small model answers first and may reply `ESCALATE` when unsure; then the large model answers. |
| `learned` strategy | Logistic regression on the question's embedding, trained on past answers, predicts whether the small model will be right; below a threshold the question goes to the large model. Standard library only (`learned.py`). |
| `consistency` strategy | The small model answers twice (the second time sampled at temperature 0.8); if the two final answers differ, the large model answers. |
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

### GSM8K: a public benchmark

The 40 questions above are mine, so a router could be tuned to them by accident. `python -m evals.gsm8k --n 500` uses problems I did not write: 500 grade-school maths word problems sampled with a fixed seed from the [GSM8K](https://github.com/openai/grade-school-math) test set (MIT licence). The first 250 train the learned router and pick its threshold; every number below is on the other 250. Each problem was asked once to each model (plus the cascade prompt and one sampled reply), every reply stored, and each strategy simulated from the stored replies. Same models and prices as above, 2026-11-26. Report: [`evals/results-gsm8k.json`](evals/results-gsm8k.json); trained router: [`evals/gsm8k-router.json`](evals/gsm8k-router.json).

| Strategy (250 test problems) | Accuracy | Sent to large model | Per million questions |
| --- | --- | --- | --- |
| Always large | 96.4% | 100% | $5,719 |
| Always small | 74.8% | 0% | $464 |
| Rules | 96.4% | 94.8% | $5,546 (−3%) |
| Cascade | 95.6% | 54.8% | $4,398 (−23%) |
| **Consistency** | **96.4%** | **26.0%** | **$2,819 (−51%)** |
| **Learned** | **95.6%** | **24.4%** | **$2,429 (−58%)** |
| Random router, same share as learned | 80.1% | 24.4% | $1,746 |
| Oracle (knows which model is right) | 96.8% | 25.2% | $2,282 (−60%) |

What the numbers mean, honestly:

- On hard problems the small model is right only 74.8% of the time, so routing matters here. **Consistency is the safe choice**: no accuracy lost, half the cost. When the small model gives the same answer twice, it is usually right.
- **The learned router beats chance by 15.5 points** at the same share of large-model calls, and it is the cheapest option within one point of the large model. It is close to the oracle's share (24.4% against 25.2%), but it still sends some wrong questions: it loses 0.8 points.
- **Rules do not work on this set**: almost every GSM8K problem is long, has three or more numbers and uses words like "each", so 94.8% go to the large model. Rules fit mixed traffic, not uniform word problems.
- **Cascade escalated more here (54.8%) but still lost 0.8 points**: a small model's own "I am unsure" is a weak signal.
- The threshold picked on the training half is 0.01, which means the trained model is very confident (250 examples, 768 features). The sweep in the report shows thresholds 0.5 to 0.9 reach 96.0% to 96.4% at 44% to 60% large-model share; more training data should give calmer probabilities.
- With 250 test problems, the 95% sampling interval is about ±2 points around 96%. One run, one seed. The stored replies (5.7 MB) are not in the repository; the command above re-creates them in about 5 hours on a Mac Studio.

## Tests

`pytest` runs 26 tests with a fake model and a fake embedding, so routing, escalation, the empty-reply fallback, cache hits, the number guard, size limit and time-to-live, metrics, cost and grading are checked without a network or a GPU.

```
src/llm_router/
  client.py   OpenAI-compatible chat and embeddings (stdlib), with token counts
  router.py   difficulty rules, strategies, escalation and fallback
  learned.py  logistic regression router (standard library)
  cache.py    semantic cache with the number guard, size limit and TTL
  metrics.py  per-run summary
  cli.py      llm-router ask
evals/        40 questions, 15 paraphrases, 5 number traps and the runner; gsm8k.py for the public benchmark
```

## Why this exists

My own AI assistant routes work between a local model and paid cloud models with spending caps. This repository is a clean, public version of that idea with the measurements in the open.

## License

MIT
