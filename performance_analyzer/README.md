---
title: Performance Analyzer
emoji: 🏆
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.15.2
python_version: '3.11'
app_file: app.py
pinned: false
---

# Performance Analyzer

**Live demo:** [https://huggingface.co/spaces/anjali209/Perf-Analyze](https://huggingface.co/spaces/anjali209/Perf-Analyze)

A tool that turns Cerebras `.xlsx` perf sweep files into actionable views for two audiences:

- **Customer / PM** — go/no-go signal against user-defined requirements, ranked throughput bar, performance frontier scatter (throughput vs TTFT), and a model × workload profile heatmap
- **Internal Engineer** — throughput distribution, relative metric comparison, scaling efficiency, config sensitivity curves, and a t-SNE cluster map

Models A–K are pre-loaded on startup. Upload additional models to compare them against the existing set. Any conforming sweep (including unseen models) renders without code changes.

---

## Install

Requires Python 3.9+.

```bash
git clone https://huggingface.co/spaces/anjali209/Perf-Analyze
cd Perf-Analyze
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` packages:

```
gradio
pandas
openpyxl
plotly
numpy
scikit-learn
```

---

## Launch

```bash
python app.py   # or: python3 app.py
```

Open the URL printed in the terminal (default: `http://127.0.0.1:7860`).

---

## Usage

### Pre-loaded models

Models A–K load automatically at startup — no upload required to see results. All charts render immediately across both tabs.

### Adding new models

1. Drag and drop one or more `.xlsx` sweep files onto the upload area.
2. Click **Add Models to Analysis**.
3. New models are highlighted across all charts; existing models A–K appear as gray reference.
4. Click **Show All Models Equally** to remove the highlight and return to a neutral view.

Files should follow the naming pattern `Model <name> profile <N>.xlsx` (e.g. `Model L profile 1.xlsx`). If the filename does not match, the app still loads the file but warns you — the model will be labelled from the filename stem and the profile will be `—`.

### Go / no-go thresholds (Customer / PM tab)

Set any combination of requirements in the **Your Performance Requirements** accordion:

| Slider | Meaning |
|---|---|
| Min Throughput (t/s) | Minimum system throughput the model must sustain |
| Min Gen Speed (t/s/user) | Minimum per-user token generation speed |
| Max TTFT (ms) | Maximum time-to-first-token the model is allowed |

Throughput and Gen Speed default to **0 (disabled)**. TTFT defaults to the dataset maximum (all rows pass — effectively disabled). Go/no-go is evaluated **only for newly uploaded models**; pre-loaded A–K are shown as gray reference bars.

A model is **GO** if it meets all active requirements in ≥ 80% of its tested configurations, **CAUTION** in 50–79%, **NO-GO** below 50%.

After adjusting sliders, click **Update Go/No-Go** to re-evaluate without re-uploading.

---

## Charts

### Customer / PM tab

**Go / No-Go banner**
Colored headline showing how many uploaded models meet the active requirements. Green = all pass, amber = partial, red = none. Below the headline, each uploaded model gets an individual badge (✅ GO / ⚠️ CAUTION / ❌ NO-GO). When no thresholds are set, the banner shows a neutral summary ranked by peak throughput.

**Ranked by Peak Throughput**
Horizontal bar chart — one bar per model, sorted from slowest to fastest. When thresholds are active, bars for uploaded models are colored by their go/no-go status; pre-loaded A–K models appear in gray as reference. Lets a PM answer "which model is fastest?" at a glance.

**Performance Frontier — Peak Throughput vs Min TTFT**
Scatter plot where each dot represents a model's best achievable point: highest throughput on the Y axis, lowest time-to-first-token on the X axis. Top-left = best tradeoff. Uploaded models are full-opacity; existing models fade to 25% so the new model's position is immediately clear.

**Peak Throughput by Model × Workload Profile**
Heatmap grid where rows are models and columns are workload profiles (labelled with input→output token counts, e.g. `P1: 1K→128`). Cell color shows peak throughput — darker blue = faster. Blank cells indicate untested combinations. Shows which model performs best under each specific workload.

---

### Engineer tab

**Throughput Distribution per Model**
Box-and-whisker plot showing the spread of throughput values across all profiles and batch sizes for each model. The box covers the middle 50% of runs; individual points are overlaid. A wide box means performance varies heavily with config; a narrow box means it is consistent. Uploaded models appear at the left with full opacity.

**Relative Performance Comparison**
Grouped bar chart where each model's score on five metrics (Throughput, TTFT, Gen Speed/user, Throughput/box, RPM) is normalized so that the best-performing model on each metric scores 100%. Makes it easy to see which model leads on which dimension and where a new model trails the field. For TTFT, a lower raw value maps to a higher score.

**Throughput Scaling Efficiency**
Line chart showing how throughput changes as batch size increases, with each line normalized to the model's throughput at batch size 1. A perfectly linear scaler would be a flat line at 100%. Lines dropping below 100% indicate sub-linear scaling. The table below the chart lists any regressions — cases where adding more concurrent requests actually reduces total throughput.

**Config Sensitivity — Throughput & Latency vs Batch Size**
Dual-panel line chart (throughput on top, latency on bottom) showing raw absolute values across the batch size sweep, one line per model/profile combination. Unlike the scaling chart, this shows actual numbers rather than normalized ratios, so engineers can read off exact throughput and latency at each operating point.

**Model Similarity (t-SNE)**
Two-dimensional cluster map produced by t-SNE — a nonlinear dimensionality reduction that projects all 14 performance metrics down to two axes. Models that behave similarly across all metrics end up close together; outliers appear far from any cluster. Each dot is a single (model, profile, batch size) data point, colored by model. If a new model was uploaded, its points appear as gold stars (★), making it immediately visible where it sits relative to known models.

---

### Upload requirements

For charts to render correctly, uploaded files must:

| Requirement | Effect if violated |
|---|---|
| `.xlsx` format | Rejected by the upload widget |
| Column names match exactly (case, spacing, parentheses) | Dependent charts silently blank |
| Header row within first 15 rows of the file | All columns misread; all charts blank |
| Metric columns contain plain numbers (no embedded units) | Those columns blank in charts |
| `Batch Size` column present | Scaling and config sensitivity charts blank |
| `Throughput (t/s)` column present | Ranked bar, box plots, pareto scatter, scaling charts blank |

The app validates each file on upload and shows a warning or error banner above the tabs if any of these conditions are violated.

---

## Dataset

The `dataset/` folder ships with 38 sweep files covering Models A–K across Profiles 1–7. These are pre-loaded at startup and used to compute slider ranges. They do not need to be re-uploaded — upload only the new models you want to compare.
