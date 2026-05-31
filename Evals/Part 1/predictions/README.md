# `Evals/Part 1/predictions/` — Full model predictions

Full prediction trajectories for the Part 1 models (LiveCodeBench v5 + AA-LCR). Shipped
so candidates can read the actual model outputs (reasoning traces, code attempts, tool
calls) for stratification, contamination checks, or anything else.

Per-sample scores ship alongside, in `Evals/Part 1/reviews/` — see
[Scores](#scores) below. You do not need to run live inference.

## Files

| File | Rows | Size |
|---|---|---|
| `live_code_bench_v5__kimi-k2.5.jsonl` | 315 | ~75 MB |
| `live_code_bench_v5__minimax-m2.5.jsonl` | 315 | ~69 MB |
| `live_code_bench_v5__gpt-oss-120b.jsonl` | 315 | ~19 MB |
| `aa_lcr__kimi-k2.5.jsonl` | 100 | ~2 MB |
| `aa_lcr__gpt-oss-120b.jsonl` | 100 | ~2 MB |
| `aa_lcr__minimax-m2.5.jsonl` | 100 | ~1 MB |

> The LiveCodeBench files are tens of MB of full trajectories — too big to open in an
> editor. Stream them line-by-line (keyed by `index`) and pull only the fields you need.

## Schema (per row)

Each row is the model's raw inference trajectory with three bulky fields stripped (see
"What was stripped" below). Top-level keys:

| Field | Type | Notes |
|---|---|---|
| `index` | int | Sample id. Aligns with `index` in the matching `Evals/Part 1/reviews/` file. |
| `model` | str | `gpt-oss-120b`, `kimi-k2.5`, or `minimax-m2.5`. |
| `model_output` | dict | The model's response. `choices[0].message.content` is the structured output (reasoning + text blocks). `usage` has token counts. |
| `messages` | list | The prompt as sent to the model. |
| `metadata` | dict | Per-sample metadata. For AA-LCR: `question`, `data_source_urls`, `input_tokens`. For LCB: typically empty (LCB metadata lives on HuggingFace `livecodebench/code_generation_lite`). |

To get the model's final text from one row:

```python
import json
rec = json.loads(line)
content = rec["model_output"]["choices"][0]["message"]["content"]
# content is a list of {"type": "reasoning"|"text", ...} blocks
text_blocks = [b["text"] for b in content if b.get("type") == "text"]
final = text_blocks[-1] if text_blocks else ""
```

## Scores

Per-sample scores ship in the sibling directory, one review file per prediction file:

```
Evals/Part 1/reviews/<benchmark>__<model>.jsonl
```

Each review row is `{index, sample_score}` and joins to a prediction row by `index`.
`sample_score.score.value` holds the metric: `pass` (0/1) for LiveCodeBench, `acc` (0/1)
for AA-LCR. All three models are graded for both benchmarks.

- **LiveCodeBench** is scored by LCB's deterministic sandbox grader.
- **AA-LCR** is scored by an LLM judge and is therefore non-deterministic — any variance
  analysis on AA-LCR partly measures judge noise. See `Task2_Model_Quality.md`.

## What's kept and what was stripped

These are near-complete inference trajectories — intentionally large. For LiveCodeBench the
**only** field removed is the per-token logprob distribution
(`model_output.choices[*].logprobs`), which was the overwhelming bulk of the raw files
(~97% — the kimi trajectory was 2.4 GB with it). Everything else is preserved: full
reasoning + text, tool calls, the tokenized prompt (`model_output.prompt_token_ids`), and
the generated `token_ids`.

The AA-LCR files are small to begin with and additionally have the token-id fields removed.

## Regeneration

Sophie regenerates this data with `python scripts/copy_predictions.py` (predictions) plus
`scripts/run_lcb_review.py` / `scripts/run_review.py` (scores). Candidates do not run
these scripts.
