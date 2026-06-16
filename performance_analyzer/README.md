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

- **Customer / PM** — go/no-go signal against user-defined latency and throughput requirements, plus a ranked bar chart and performance frontier scatter
- **Internal Engineer** — throughput distribution, relative metric comparison, scaling efficiency, config sensitivity curves, and a t-SNE / PCA explorer

Upload one file or many; comparison across models is the default, not an afterthought. Any conforming sweep (including unseen models) renders without code changes.

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

`requirements.txt` pins the following packages:

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

### Analyzing uploaded sweeps

1. Drag and drop one or more `.xlsx` perf sweep files onto the upload area.  
   Files should follow the naming pattern `Model <X> profile <N>.xlsx`.
2. Click **Analyze Uploaded Files**.
3. Results appear in the **Customer / PM** and **Engineer** tabs.

### Go / no-go thresholds (Customer / PM tab)

Set any combination of requirements in the **Your Performance Requirements** accordion:

| Slider | Meaning |
|---|---|
| Min Throughput (t/s) | Minimum system throughput the model must sustain |
| Min Gen Speed (t/s/user) | Minimum per-user token generation speed |
| Max TTFT (ms) | Maximum time-to-first-token the model is allowed |

Each slider's range spans the weakest to strongest model median in the dataset — hover the ℹ icon for details. A model is **GO** if it meets all active requirements in ≥ 80% of its tested configurations, **CAUTION** in 50–79%, **NO-GO** below 50%.

After adjusting sliders, click **Update Go/No-Go** to re-evaluate without re-uploading.

### PCA / t-SNE explorer (Engineer tab)

Click **Run Analysis on Full Dataset** inside the *PCA / t-SNE Explorer* accordion. If files have been uploaded first, any model not already in `dataset/` is overlaid on the cluster map as a **gold star** marker so you can see immediately where an unseen model sits relative to known ones.

---

## Dataset

The `dataset/` folder ships with 38 sweep files covering Models A–K across Profiles 1–7. These are used automatically by the PCA explorer and to compute slider ranges at startup. They do not need to be re-uploaded for the main analysis — upload only the files you want to compare.
