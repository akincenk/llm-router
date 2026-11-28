# llm-router

[![CI](https://github.com/akincenk/llm-router/actions/workflows/ci.yml/badge.svg)](https://github.com/akincenk/llm-router/actions/workflows/ci.yml)

Send each question to the cheapest model that can answer it, and skip the model entirely when the same question was already answered. Standard library only; works with any OpenAI-compatible endpoint, including fully local models through [Ollama](https://ollama.com).

On a 40-question test set, routing by explainable rules kept the accuracy of always using the large model (**100%**) at **58% lower cost**. A semantic cache in front answered **14 of 15** reworded questions without calling any model and was **never fooled by changed numbers (0 of 5)**.

On the cleaned **GSM8K-Platinum** maths benchmark (1,209 problems, 2026-11-27), with the small model's thinking turned off, **three-sample self-consistency reached 99.1% against the large model's 99.3% at 90% lower cost**, sending 2.6% of problems to the large model. The learned router was no better than a random router at the same share, and question-only signals predicted nothing (AUROC about 0.5), so the routing decision that works is the small model disagreeing with itself.

History: on 2026-11-26 (GSM8K v1, 250 problems, small model thinking on) consistency matched the large model's 96.4% at 51% lower cost; a re-analysis found that the learned router's result there came from the empty-reply fallback, not from learning. Both versions are below.

## How it works

| Part | What it does |
| --- | --- |
| `rules` strategy | An explainable difficulty score (length, 3+ numbers, reasoning words such as "each", "after", "probability", several questions, code). Two or more signals send the question to the large model; every answer records why. |
| `cascade` strategy | The small model answers first and may reply `ESCALATE` when unsure; then the large model answers. |
| `learned` strategy | Logistic regression on the question's embedding, trained on past answers, predicts whether the small model will be right; below a threshold the question goes to the large model. Standard library only (`learned.py`). |
| `consistency` strategy | The small model answers twice (the second time sampled at temperature 0.8); if the two final answers differ, the large model answers. The GSM8K evaluation also simulates `consistency3` (three answers must all agree), `majority3` (two of three agreeing is enough, never escalates on a 2-1 vote) and `learned_consistency` (the learned router first, the consistency check only on the questions it calls easy). |
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

### GSM8K v2: GSM8K-Platinum, small model thinking off (2026-11-27)

The first GSM8K run (next section) had two flaws: 58 of 250 small-model answers were empty because Gemma 4 thought until its token budget ran out, and some GSM8K labels are wrong. v2 changes four things:

1. **Thinking off for the small model** (`reasoning_effort: none` through the new per-model `Client(options=...)`). Small-model replies averaged 173 output tokens and 1.7 seconds; none of its 1,709 replies was empty.
2. **An `Answer:` line**: the system prompt asks for a last line `Answer: <final answer>`, and grading takes the first number after it, so "Answer: 180,000 m (180 km)" reads as 180000, not 180.
3. **Cleaner data**: training on 500 problems from the GSM8K *train* file, testing on all 1,209 problems of [GSM8K-Platinum](https://github.com/MadryLab/platinum-benchmarks) (MadryLab: GSM8K test problems with label errors fixed and ambiguous problems removed, MIT licence), so test and training never overlap.
4. **The checklist of [arXiv 2610.07354](https://arxiv.org/abs/2610.07354)** ("Evaluating Escalation Signals for LLM Routing: Targets, Controls, and Five Ways to Fool Yourself", 2026-10-05): a fallback-only control row, question-only signals as a control for every live signal, two targets ("small model wrong" and "escalating helps": small wrong and large right), the headroom left above the large model, and live sampling counted in the cost. The learned router's L2 strength is picked by cross-validation and its threshold on out-of-fold predictions, both on the training set only.

Same models and prices as above, Ollama 0.40.18, one run (2026-11-26 22:37 to 2026-11-27 05:55, times as logged). Report: [`evals/results-gsm8k-v2.json`](evals/results-gsm8k-v2.json); trained router: [`evals/gsm8k-v2-router.json`](evals/gsm8k-v2-router.json). Re-running the analysis from the stored replies gave a byte-identical report.

| Strategy (1,209 test problems) | Accuracy | Correct | Sent to large model | Per million questions |
| --- | --- | --- | --- | --- |
| Always large | 99.34% | 1,201 | 100% | $5,445 |
| Always small | 98.26% | 1,188 | 0% | $96 (−98%) |
| Empty-reply fallback only (control) | 98.26% | 1,188 | 0% | $96 (−98%) |
| Rules | 99.34% | 1,201 | 95.2% | $5,278 (−3%) |
| Cascade | 98.18% | 1,187 | 0% | $96 (−98%) |
| Consistency (2 samples) | 98.92% | 1,196 | 2.2% | $404 (−93%) |
| **Consistency3 (3 samples must agree)** | **99.09%** | **1,198** | **2.6%** | **$528 (−90%)** |
| Majority3 (2 of 3 is enough) | 98.18% | 1,187 | 0% | $289 (−95%) |
| Learned | 98.35% | 1,189 | 14.8% | $957 (−82%) |
| Learned + consistency | 98.92% | 1,196 | 2.2% | $404 (−93%) |
| Random router, same share as learned | 98.42% | | 14.8% | $888 |
| Oracle (knows which model is right) | 99.75% | 1,206 | 1.7% | $269 (−95%) |

| Signal: AUROC for… (1,209 test problems; 21 small-model errors, 18 where escalating helps) | small model wrong | escalating helps |
| --- | --- | --- |
| Question only: rules signal count (control) | 0.54 | 0.50 |
| Question only: learned 1 − p(correct) (control) | 0.49 | 0.51 |
| Live: the 2 small-model samples disagree | 0.68 | 0.71 |
| Live: disagreement among 3 samples | 0.73 | 0.77 |
| Live: cascade `ESCALATE` | 0.50 | 0.50 |

What the numbers mean, honestly:

- **Turning thinking off fixed the small model, not the router.** Gemma 4 went from 74.8% (v1, many empty replies) to 98.26% here, only 13 problems behind the large model. That leaves almost no room: the oracle is just 0.41 points above always using the large model (the paper's "headroom" check).
- **Consistency3 is the best trade-off**: 3 problems fewer than the large model (1,198 against 1,201) for a tenth of the price. Disagreement among the small model's own samples is the only signal that beats the question-only controls (0.73 / 0.77 against about 0.5).
- **The learned router did not learn.** Its question-only probability scores AUROC 0.49, and it is slightly *worse* than a random router at the same 14.8% share (98.35% against 98.42%). The fallback control shows the fallback did nothing this time (0 empty replies), so v1's correction holds: the earlier learned result came from the fallback.
- **The cascade never escalated** (0 of 1,209): without thinking, the small model never says it is unsure. **Majority3 never escalated either**, because 2-1 votes are accepted; it adds sampling cost for no accuracy.
- **Rules get the top accuracy only by sending 95% of problems to the large model** (−3% cost), as in v1: every GSM8K problem looks hard to them.
- Platinum's cleaner labels and the thinking switch both raise accuracy, so v2 numbers are not comparable with v1's; v1 is kept below as history. With 1,209 problems, differences of 3 or 4 problems (0.25 to 0.33 points) are within noise. One run, one seed. The stored replies (22 MB) are not in the repository; the command `python -m evals.gsm8k --train 500 --cache evals/gsm8k-v2-replies.json` re-creates them (this run took about 7 hours 20 minutes on a Mac Studio).

### GSM8K v1 (2026-11-26, history)

The 40 questions above are mine, so a router could be tuned to them by accident. `python -m evals.gsm8k --n 500` uses problems I did not write: 500 grade-school maths word problems sampled with a fixed seed from the [GSM8K](https://github.com/openai/grade-school-math) test set (MIT licence). The first 250 train the learned router and pick its threshold; every number below is on the other 250. Each problem was asked once to each model (plus the cascade prompt and one sampled reply), every reply stored, and each strategy simulated from the stored replies. Same models and prices as above, 2026-11-26. Report: [`evals/results-gsm8k.json`](evals/results-gsm8k.json); trained router: [`evals/gsm8k-router.json`](evals/gsm8k-router.json).

| Strategy (250 test problems) | Accuracy | Sent to large model | Per million questions |
| --- | --- | --- | --- |
| Always large | 96.4% | 100% | $5,719 |
| Always small | 74.8% | 0% | $464 |
| Rules | 96.4% | 94.8% | $5,546 (−3%) |
| Cascade | 95.6% | 54.8% | $4,398 (−23%) |
| **Consistency** | **96.4%** | **26.0%** | **$2,819 (−51%)** |
| Learned (see the correction below) | 95.6% | 24.4% | $2,429 (−58%) |
| Empty-reply fallback only, no routing decision | 95.6% | 23.2% | $2,392 (−58%) |
| Random router, same share as learned, no fallback | 80.1% | 24.4% | $1,746 |
| Oracle (knows which model is right) | 96.8% | 25.2% | $2,282 (−60%) |

What the numbers mean, honestly:

- On hard problems the small model is right only 74.8% of the time, so routing matters here. **Consistency is the safe choice**: no accuracy lost, half the cost. When the small model gives the same answer twice, it is usually right.
- **Correction (2026-11-26, found by re-reading the stored replies): the learned router did not learn much.** 58 of the 250 small-model answers were empty: Gemma 4 thought until its 2,048-token budget ran out. Every strategy sends an empty reply to the large model, and that fallback alone gives 95.6% at 23.2% large share. The trained model added 3 escalations and no accuracy. The random router I compared against had no fallback, so "15.5 points better than random", which this section said first, measured the fallback, not the model. As a signal for "the small model is wrong", the learned probability scores AUROC 0.67, close to the simple rules count (0.65); the disagreement of two small-model samples scores 0.76. The next measurement turns thinking off for the small model, uses the cleaned GSM8K-Platinum labels, and reports a fallback-only control and signal AUROCs next to every strategy.
- **Rules do not work on this set**: almost every GSM8K problem is long, has three or more numbers and uses words like "each", so 94.8% go to the large model. Rules fit mixed traffic, not uniform word problems.
- **Cascade escalated more here (54.8%) but still lost 0.8 points**: a small model's own "I am unsure" is a weak signal.
- The threshold picked on the training half is 0.01, which means the trained model is very confident (250 examples, 768 features). The sweep in the report shows thresholds 0.5 to 0.9 reach 96.0% to 96.4% at 44% to 60% large-model share; more training data should give calmer probabilities.
- With 250 test problems, the 95% sampling interval is about ±2 points around 96%. One run, one seed. The stored replies (5.7 MB) are not in the repository; the command above re-creates them in about 5 hours on a Mac Studio.

## Tests

`pytest` runs 32 tests with a fake model and a fake embedding, so routing, escalation, the empty-reply fallback, cache hits, the number guard, size limit and time-to-live, metrics, cost, grading (including the `Answer:` line), the three-sample strategies, the fallback control and AUROC are checked without a network or a GPU.

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
