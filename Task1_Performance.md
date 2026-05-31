# Task 1 — Performance UI for Customer and Product

## The scenario

Cerebras is known for being fast. Customers care about throughput and gen speed — they want to know, before they commit, whether a model running on us will keep up with their workload.

Today, that answer lives in an `.xlsx` perf projection sheet that only Cerebras perf engineers can read. A customer-facing PM has to open the sheet, decode the column names, and translate the numbers into something the customer can act on. The customer never sees the file directly.

Build the layer that turns those projections into an answer two different audiences can act on:

- **Customer / customer-facing PM** — needs a go/no-go signal: is the model fast enough for the workload they described? What does "fast" look like in numbers they recognize (tok/s, TTFT, context, cost implications)?
- **Internal product / deployment engineer** — needs to sanity-check a projection, see how it shifts with config, and spot anomalies before they reach a customer.

You decide how to serve both audiences. One UI with two views, two separate UIs, or a defended argument for which audience to build for first — all defensible, none obvious.


## The challenge

Build a UI that lets the two audiences above answer their questions from the shipped data. You decide what data to provide and what UI to use

## Hard requirements

- **One-command launch**. `pip install -e ".[dev]"` plus one launch command brings up the UI. Document both in your README.
- **Defensible for a twelfth model**. We will mentally substitute a `Model L` (or drop in a new `Model_L_profile_<N>/Model L profile <N>.xlsx`) and see if your design still works. Hard-coded model lists or hand-tuned views that only work for the shipped letters will be marked down.


## Forbidden trivial baselines

These will not pass the rubric. We will check.

- Static HTML dump of the `.xlsx` or a single big table with no audience differentiation.
- A UI that shows every column to every audience and calls it "configurable."
- Hard-coded views or copy that only work for the specific models in `perf_data.zip`.

## What to submit

Two things.

### 1. Code

A link to (or zip of) your project. Your project must run this two-command contract:

```bash
# Install + sanity check shipped data
pip install -e ".[dev]" && python scripts/verify_perf_data.py

# Launch the UI (your choice of framework)
<your-launch-command>
```

### 2. Video Explanation (5 min max)
Create a video where you explain
- What problem did you understand yourself to be solving for each audience? What did you cut and why?
- What framework did you pick and what did you rule out?
- What assumptions did you make about the data, the audience, and the deployment context?
- What would change if you had (a) more data, (b) production measurements alongside the projections, (c) more time?
- Based on the performance numbers what size do you think each model A-K represent
- Based on the traffic profile, what do you think are the use case of profile 1-7
