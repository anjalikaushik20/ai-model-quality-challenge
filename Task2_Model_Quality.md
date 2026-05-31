# Task 2 — Benchmark Compression for a Real Customer

## The scenario

A Cerebras prospect is evaluating us for production deployment. Their product depends on two model capabilities: **code generation** and **long-context reasoning**. They want a fast answer to one question: *is this model good enough for our workload?*

Running our full benchmark suite across the candidate models is extremely expensive.

Help us prune these benchmarks to the smallest sample set that still gives a useful good-or-not signal for this customer.

Sales is also asking what this customer's roadmap might look like if they extend into multimodal next quarter. Sketch how you would probe that capability cheaply if it comes up — see Part B.

## Datasets shipped

The repo ships pre-computed model outputs under `Evals/<part>/predictions/` and the
matching per-sample scores under the sibling `Evals/<part>/reviews/` directory
(`Evals/Part 1/` for LiveCodeBench + AA-LCR, `Evals/MMMU/` for MMMU). Predictions and
reviews join on the `index` field. Per-sample metadata is embedded in the rows
themselves (in the prediction `metadata` field and the review `sample_score`), not in a
separate file. You do not need to run live inference.

| Benchmark | Customer capability | Samples shipped | Models | Score source |
|---|---|---|---|---|
| LiveCodeBench v5 | Coding | 315 | `gpt-oss-120b`, `kimi-k2.5`, `minimax-m2.5` | LCB sandbox grader (`pass`) |
| AA-LCR | Long-context reasoning | 100 | `gpt-oss-120b`, `kimi-k2.5`, `minimax-m2.5` | LLM judge (`acc`) |
| MMMU | Forward-looking multimodal | 660 reference samples (22 subjects × 30) | `glm-4.5v-fp8` | LLM judge (`acc`) |

For Part B you must reason about **the full MMMU dataset (~12K samples on HuggingFace: `MMMU/MMMU`)**, not only the 660 reference rows. The reference rows are illustrative of one model's behavior.

> **Note on AA-LCR scoring**: AA-LCR is graded by an LLM judge, which is non-deterministic. Any variance analysis you do on AA-LCR will partly measure judge noise, not sample variance. Account for this if you can.

## The challenge

You have two parts. Depth on Part A beats shallow coverage of both. It is acceptable to do Part A deeply and treat Part B as a written design proposal in Handout A.

### Part A — Coding + long-context capability compression (LCB v5 + AA-LCR)

Prune each benchmark in a way that still tells our sales team whether a model is good enough for this customer and for engineering to use in our testing.


### Part B — Forward-looking multimodal probe (MMMU, full ~12K HF dataset)

If this customer adds multimodal next quarter, how would we cheaply tell them whether a candidate model's image encoder is good enough?

There are data samples of MMMU, however do not be limited by just the answers in the model output. 

Design a pruning strategy that selects a probe set for the full 12K MMMU dataset. The probe should surface image-encoder degradation specifically — not generic capability gaps. What kinds of images stress an encoder? How would you measure encoder quality given you must interact with the model through the standard OpenAI interface. 


## Hard requirement — your pruner must live inside `evalscope`

This is not a standalone-script exercise. Your pruning code must be implemented as an extension to the public [`modelscope/evalscope`](https://github.com/modelscope/evalscope) codebase. Treat it like a PR you would submit upstream: pick a clean extension point, follow the framework's conventions, and write code an evalscope maintainer would not reject on style alone.


**Pin the evalscope commit SHA you developed against** in your fork's README. The framework's APIs are still evolving.

## Forbidden trivial baselines

These will not pass the rubric. We will check.

- Uniform random sampling.
- Top-k easiest or top-k hardest by score.
- Hand-picked samples.
- Strategies that overfit to the three shipped models — your method should be defensible for a fourth model we have not given you.

## What to submit

Three things.

### 1. Code

A link to (or zip of) your evalscope fork. Your fork must run the three-command contract end to end:

```bash
# Reference run
evalscope eval --model <model> --datasets live_code_bench --output ./results_full/

# Pruned run (your pruner)
evalscope eval --model <model> --datasets live_code_bench_pruned \
    --dataset-args '{"pruning_strategy": "your_strategy", "prune_ratio": 0.1}' \
    --output ./results_pruned/

# Compare
python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/
```

### 2. Handout A — "Why this works" (1 page, technical audience)

Written for an engineer who could have built this themselves. Cover:

- Why you chose this approach for Part A. What problem did you understand yourself to be solving?
- What you would do for Part B and why those choices stress image encoders specifically.
- What assumptions you made about distribution, scale, and model behavior.
- What would change if you had (a) more data, (b) a live model endpoint to query, or (c) more time?

### 3. Handout B — "Why this matters and how to use it" (1 page, mixed audience: developers, test engineers, product, customer team)

Written for someone who does not work on evaluation. Cover:

- What does shipping these pruners change for the customer conversation in concrete terms?
- How would a sales engineer or deployment lead actually *run* this tomorrow inside evalscope to get a go/no-go answer for a customer?
- What signal does the multimodal probe give that random sampling cannot?
- Why should a customer-facing PM care about either of these?

This handout is not a watered-down version of Handout A. The audience is different. Translate, do not dumb down.
