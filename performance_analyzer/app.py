import glob
import gradio as gr
import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
METRIC_COLS = [
    "Max number of milliseconds",
    "Target Max number of milliseconds",
    "Prompt only Throughput (t/s)",
    "Gen only Throughput (t/s)",
    "Throughput (t/s)",
    "Throughput / box (t/s/hardware)",
    "Uncached Throughput (t/s)",
    "Uncached Throughput / box (t/s/hardware)",
    "Cached Throughput (t/s)",
    "Cached Throughput / box (t/s/hardware)",
    "TTFT (ms)",
    "Real Prompt Speed (t/s/user)",
    "Prompt Speed with Queueing (t/s/user)",
    "Gen Speed (t/s/user)",
    "RPM",
]
CONFIG_COLS = ["Input Length", "Output Length", "Cache %", "Batch Size"]

PCA_FEATURE_COLS = [
    "Max number of milliseconds",
    "Prompt only Throughput (t/s)",
    "Gen only Throughput (t/s)",
    "Throughput (t/s)",
    "Throughput / box (t/s/hardware)",
    "Uncached Throughput (t/s)",
    "Uncached Throughput / box (t/s/hardware)",
    "Cached Throughput (t/s)",
    "Cached Throughput / box (t/s/hardware)",
    "TTFT (ms)",
    "Real Prompt Speed (t/s/user)",
    "Prompt Speed with Queueing (t/s/user)",
    "Gen Speed (t/s/user)",
    "RPM",
]

THROUGHPUT_COL = "Throughput (t/s)"
LATENCY_COL    = "Max number of milliseconds"
TARGET_COL     = "Target Max number of milliseconds"
TTFT_COL       = "TTFT (ms)"
RPM_COL        = "RPM"

PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    "#FF97FF", "#FECB52", "#72B7B2", "#E45756",
]

STATUS_EMOJI  = {"GO": "✅ GO", "CAUTION": "⚠️ CAUTION", "NO-GO": "❌ NO-GO"}
STATUS_COLORS = {
    "GO":      ("#2d6a2d", "#e8f5e9"),
    "CAUTION": ("#7a5c00", "#fff8e1"),
    "NO-GO":   ("#8b1a1a", "#ffebee"),
}

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


# ---------------------------------------------------------------------------
# File ingestion
# ---------------------------------------------------------------------------

def _find_header_row(path, max_scan=15):
    known = set(METRIC_COLS + CONFIG_COLS)
    preview = pd.read_excel(path, header=None, nrows=max_scan)
    for i, row in preview.iterrows():
        if {str(v).strip() for v in row if pd.notna(v)} & known:
            return i
    return 0


def get_run_label(filename):
    """Derive (model, profile) from filename; unknown models fall back to the stem."""
    stem      = os.path.splitext(os.path.basename(filename))[0].replace("_", " ")
    model_m   = re.search(r"Model\s*([A-Za-z0-9]+)", stem, re.I)
    profile_m = re.search(r"profile\s*(\d+)",         stem, re.I)
    model     = f"Model {model_m.group(1)}"     if model_m   else stem
    profile   = f"Profile {profile_m.group(1)}" if profile_m else "—"
    return model, profile


_KNOWN_COLS = set(METRIC_COLS + CONFIG_COLS)


def _validate_uploaded_file(df, filename, model, profile):
    """Return (errors, warnings) for a single parsed upload frame.

    errors   → file is unusable and will be skipped
    warnings → file is included but some visuals will be degraded
    """
    errors   = []
    warnings = []
    basename = os.path.basename(filename)

    recognized = [c for c in df.columns if c in _KNOWN_COLS]
    if not recognized:
        errors.append(
            f"<b>{basename}</b>: No recognized column names found. "
            "Verify that column headers exactly match the expected names "
            "(case-sensitive, including spaces and parentheses such as "
            "<code>Throughput (t/s)</code>) and appear within the first "
            "15 rows of the file."
        )
        return errors, warnings  # no point checking further

    stem = os.path.splitext(basename)[0].replace("_", " ")
    if not re.search(r"Model\s*[A-Za-z0-9]+", stem, re.I):
        warnings.append(
            f"<b>{basename}</b>: Filename does not contain a "
            "<code>Model &lt;name&gt;</code> pattern — model will be "
            f"labelled <i>{model}</i>. "
            "Rename to e.g. <code>Model L profile 1.xlsx</code> for cleaner labels."
        )
    if not re.search(r"profile\s*\d+", stem, re.I):
        warnings.append(
            f"<b>{basename}</b>: Filename does not contain a "
            "<code>profile &lt;N&gt;</code> pattern — profile will be "
            f"labelled <i>{profile}</i>. "
            "Uploading multiple files for the same model without profile numbers "
            "may cause rows to be merged during deduplication."
        )
    if THROUGHPUT_COL not in df.columns:
        warnings.append(
            f"<b>{basename}</b>: Missing column <code>{THROUGHPUT_COL}</code> — "
            "ranked bar, box plots, pareto scatter, and scaling chart will be blank for this model."
        )
    if "Batch Size" not in df.columns:
        warnings.append(
            f"<b>{basename}</b>: Missing column <code>Batch Size</code> — "
            "scaling efficiency and config sensitivity charts will be blank."
        )
    if TTFT_COL not in df.columns:
        warnings.append(
            f"<b>{basename}</b>: Missing column <code>{TTFT_COL}</code> — "
            "the Max TTFT threshold will have no effect on this model's go/no-go result."
        )

    bad_cols = [
        c for c in recognized
        if c in METRIC_COLS and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if bad_cols:
        cols_str = ", ".join(f"<code>{c}</code>" for c in bad_cols)
        warnings.append(
            f"<b>{basename}</b>: Non-numeric values detected in {cols_str}. "
            "Those columns will be blank in charts. "
            "Ensure cells contain plain numbers with no embedded units or text."
        )

    return errors, warnings


def _build_validation_html(errors, warnings):
    """Render per-file upload errors and warnings as a styled HTML block."""
    if not errors and not warnings:
        return ""
    # Force dark text on all child elements regardless of Gradio theme overrides
    style = (
        '<style>'
        '.va-box, .va-box b, .va-box strong, .va-box i, .va-box li, .va-box span'
        ' { color: #1a1a1a !important; }'
        '.va-box code { color: #1a1a1a !important; background: rgba(0,0,0,0.06) !important;'
        ' padding: 1px 4px; border-radius: 3px; }'
        '</style>'
    )
    parts = [style]
    if errors:
        items = "".join(f"<li style='margin:4px 0'>{e}</li>" for e in errors)
        parts.append(
            '<div class="va-box" style="background:#ffebee;border-left:4px solid #c62828;'
            'padding:12px 16px;border-radius:4px;margin-bottom:8px;">'
            '<strong style="color:#b71c1c !important;">Upload errors — the following files were skipped:</strong>'
            f'<ul style="margin:6px 0 0 16px;padding:0">{items}</ul></div>'
        )
    if warnings:
        items = "".join(f"<li style='margin:4px 0'>{w}</li>" for w in warnings)
        parts.append(
            '<div class="va-box" style="background:#fff8e1;border-left:4px solid #f9a825;'
            'padding:12px 16px;border-radius:4px;margin-bottom:8px;">'
            '<strong style="color:#e65100 !important;">Upload warnings — some charts may be incomplete:</strong>'
            f'<ul style="margin:6px 0 0 16px;padding:0">{items}</ul></div>'
        )
    return "".join(parts)


def read_files(files):
    """Parse uploaded xlsx files and validate each one.

    Returns (combined_df_or_None, errors, warnings).
    Files that fail the hard column check are skipped; others are merged.
    """
    frames       = []
    all_errors   = []
    all_warnings = []
    for f in files:
        model, profile = get_run_label(f.name)
        hrow = _find_header_row(f.name)
        df   = pd.read_excel(f.name, header=hrow)
        df.columns = df.columns.str.strip()
        df["Model"]   = model
        df["Profile"] = profile
        errors, warnings = _validate_uploaded_file(df, f.name, model, profile)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        if not errors:
            frames.append(df)
    if not frames:
        return None, all_errors, all_warnings
    return pd.concat(frames, ignore_index=True), all_errors, all_warnings


def load_dataset_files():
    paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.xlsx")))
    if not paths:
        return None, f"No .xlsx files found in {DATASET_DIR}"
    frames = []
    for p in paths:
        model, profile = get_run_label(p)
        hrow = _find_header_row(p)
        df   = pd.read_excel(p, header=hrow)
        df.columns = df.columns.str.strip()
        df["Model"]   = model
        df["Profile"] = profile
        frames.append(df)
    return pd.concat(frames, ignore_index=True), None


def _compute_slider_stats():
    """Load dataset once at startup to derive slider bounds and lenient/strict thresholds.

    'Lenient' = threshold where ~80% of models would be GO (p20 of per-model medians for
    'higher is better' metrics; p80 for 'lower is better').
    'Strict'  = threshold where ~20% of models would be GO (p80 / p20 respectively).
    This gives the customer a meaningful operating range rather than raw row percentiles.
    """
    df, err = load_dataset_files()
    if err or df is None:
        return {}
    result = {}
    specs = [
        (THROUGHPUT_COL,          "higher"),
        (TTFT_COL,                "lower"),
        ("Gen Speed (t/s/user)",  "higher"),
    ]
    for col, direction in specs:
        if col not in df.columns:
            continue
        vals          = df[col].dropna()
        model_medians = df.groupby("Model")[col].median()
        if direction == "higher":
            lenient = float(model_medians.quantile(0.20))  # 80% of models above this
            strict  = float(model_medians.quantile(0.80))  # 20% of models above this
        else:
            lenient = float(model_medians.quantile(0.80))  # 80% of models below this
            strict  = float(model_medians.quantile(0.20))  # 20% of models below this
        result[col] = {
            "min":               float(vals.min()),
            "max":               float(vals.max()),
            "model_min_median":  float(model_medians.min()),
            "model_max_median":  float(model_medians.max()),
            "lenient":           lenient,
            "strict":            strict,
            "direction":         direction,
        }
    return result

_SLIDER_STATS = _compute_slider_stats()
_BASE_DF, _BASE_DF_ERR = load_dataset_files()


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def present_metric_cols(df):
    return [c for c in METRIC_COLS if c in df.columns]


def detect_sweep_col(df):
    for col in ["Batch Size", "Input Length", "Output Length", "Cache %"]:
        if col in df.columns and df[col].nunique() > 1:
            return col
    return None


def detect_primary_metric(df):
    for col in [THROUGHPUT_COL, "Gen only Throughput (t/s)", "Cached Throughput (t/s)"]:
        if col in df.columns:
            return col, "higher"
    for col in [LATENCY_COL, TTFT_COL]:
        if col in df.columns:
            return col, "lower"
    available = present_metric_cols(df)
    return (available[0], "higher") if available else (None, None)


def mp_label(df):
    return df["Model"] + " / " + df["Profile"]


# ---------------------------------------------------------------------------
# Customer / PM — latency headroom logic
# ---------------------------------------------------------------------------

def _profile_label_map(df):
    """Build a dict mapping 'Profile N' -> 'PN: xK→y' using Input/Output Length from data."""
    mapping = {}
    if "Input Length" not in df.columns or "Output Length" not in df.columns:
        return mapping
    for profile, grp in df.groupby("Profile"):
        inp = grp["Input Length"].dropna().median()
        out = grp["Output Length"].dropna().median()
        if pd.isna(inp) or pd.isna(out):
            continue
        inp_k = inp / 1000
        inp_str = f"{int(inp_k)}K" if inp_k == int(inp_k) else f"{inp_k:.1f}K"
        if out >= 1000:
            out_k = out / 1000
            out_str = f"{int(out_k)}K" if out_k == int(out_k) else f"{out_k:.1f}K"
        else:
            out_str = str(int(out))
        m = re.search(r"\d+", str(profile))
        p_num = m.group(0) if m else str(profile)
        mapping[profile] = f"P{p_num}: {inp_str}→{out_str}"
    return mapping


def compute_ranking(df, thresholds=None):
    """One row per (Model, Profile) with latency headroom, peak metrics, and optional status."""
    # thresholds: dict of {col: value} where value is the customer's requirement
    # TTFT is "lower is better"; all others are "higher is better"
    active = {k: v for k, v in (thresholds or {}).items() if v is not None and v > 0 and k in df.columns}
    has_target = TARGET_COL in df.columns and LATENCY_COL in df.columns
    has_tp     = THROUGHPUT_COL in df.columns
    has_rpm    = RPM_COL in df.columns
    has_ttft   = TTFT_COL in df.columns
    sweep      = detect_sweep_col(df)

    df = df.copy()
    if has_target:
        df["_headroom"] = (
            (df[TARGET_COL] - df[LATENCY_COL]) / df[TARGET_COL].clip(lower=1)
        ) * 100

    # Compute per-row pass against each active threshold
    pass_cols = []
    if active:
        for col, thresh in active.items():
            pname = f"_pass_{col}"
            df[pname] = df[col] <= thresh if col == TTFT_COL else df[col] >= thresh
            pass_cols.append(pname)
        df["_all_pass"] = df[pass_cols].all(axis=1)

    rows = []
    for (model, profile), grp in df.groupby(["Model", "Profile"], sort=False):
        r = {"Model": model, "Profile": profile}
        if has_tp:
            peak_idx = grp[THROUGHPUT_COL].idxmax()
            r["Peak Throughput (t/s)"] = round(grp[THROUGHPUT_COL].max(), 1)
            r["Avg Throughput (t/s)"]  = round(grp[THROUGHPUT_COL].mean(), 1)
            if sweep and sweep in grp.columns:
                r[f"Peak at {sweep}"] = grp.loc[peak_idx, sweep]
        if LATENCY_COL in df.columns:
            r["Avg Latency (ms)"] = round(grp[LATENCY_COL].mean(), 1)
            r["Min Latency (ms)"] = round(grp[LATENCY_COL].min(), 1)
        if has_ttft:
            r["Min TTFT (ms)"] = round(grp[TTFT_COL].min(), 2)
        if has_rpm:
            r["Avg RPM"] = round(grp[RPM_COL].mean(), 1)
        if has_target:
            r["Avg Headroom (%)"] = round(grp["_headroom"].mean(), 2)
            r["Min Headroom (%)"] = round(grp["_headroom"].min(), 2)
        if pass_cols:
            pass_pct = round(grp["_all_pass"].mean() * 100, 1)
            r["_pass_pct"] = pass_pct
            r["Status"] = "GO" if pass_pct >= 80 else "CAUTION" if pass_pct >= 50 else "NO-GO"
        rows.append(r)

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking

    if "Status" in ranking.columns:
        order = {"GO": 0, "CAUTION": 1, "NO-GO": 2}
        ranking["_ord"] = ranking["Status"].map(order)
        sort_cols, asc = ["_ord"], [True]
        if has_tp:
            sort_cols.append("Peak Throughput (t/s)"); asc.append(False)
        ranking = ranking.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
        ranking = ranking.drop(columns=["_ord", "_pass_pct"])
    elif has_tp:
        ranking = ranking.sort_values("Peak Throughput (t/s)", ascending=False).reset_index(drop=True)

    return ranking


def build_summary_html(ranking_df, thresholds=None):
    """Go/no-go banner: threshold-based when requirements are set, neutral info otherwise."""
    if ranking_df is None or ranking_df.empty:
        return "<p>No data to display.</p>"

    active     = {k: v for k, v in (thresholds or {}).items() if v is not None and v > 0}
    has_status = "Status" in ranking_df.columns and bool(active)
    n_models   = ranking_df["Model"].nunique()

    if has_status:
        order = {"GO": 0, "CAUTION": 1, "NO-GO": 2}
        model_status = {}
        for _, row in ranking_df.iterrows():
            m, s = row["Model"], row["Status"]
            if m not in model_status or order[s] > order[model_status[m]]:
                model_status[m] = s

        go_n = sum(1 for s in model_status.values() if s == "GO")
        if go_n == n_models:
            bg, headline = "#2d6a2d", f"✅ All {n_models} models meet your requirements"
        elif go_n > 0:
            bg, headline = "#7a5c00", f"⚠️ {go_n} of {n_models} models meet your requirements"
        else:
            bg, headline = "#8b1a1a", f"❌ No models meet your requirements"

        thresh_parts = []
        THRESH_LABELS = {
            THROUGHPUT_COL:         "min throughput",
            "Gen Speed (t/s/user)": "min gen speed",
            TTFT_COL:               "max TTFT",
        }
        for col, val in active.items():
            lbl = THRESH_LABELS.get(col, col)
            unit = "ms" if col == TTFT_COL else "t/s"
            thresh_parts.append(f"{lbl}&nbsp;{val:,.0f}&nbsp;{unit}")
        sub = "Requirements:&nbsp;" + ",&nbsp;".join(thresh_parts) if thresh_parts else ""

        sub_html = (
            f'<br><span style="font-size:0.6em;font-weight:normal;opacity:0.85;">{sub}</span>'
            if sub else ""
        )
        html = (
            f'<div style="background:{bg};color:#fff;padding:16px 20px;border-radius:8px;'
            f'font-size:1.25em;font-weight:bold;margin-bottom:12px;">'
            f'{headline}{sub_html}'
            f'</div>'
        )

        html += '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">'
        for model, status in model_status.items():
            tc, bc = STATUS_COLORS[status]
            html += (
                f'<span style="background:{bc};color:{tc};border:1.5px solid {tc};'
                f'padding:4px 12px;border-radius:20px;font-size:0.9em;font-weight:600;">'
                f'{STATUS_EMOJI[status]}&nbsp;—&nbsp;{model}'
                f'</span>'
            )
        html += "</div>"

    else:
        html = (
            f'<div style="background:#1e3a5f;color:#fff;padding:16px 20px;border-radius:8px;'
            f'font-size:1.25em;font-weight:bold;margin-bottom:12px;">'
            f'{n_models} model{"s" if n_models > 1 else ""} loaded — ranked by peak throughput'
            f'<br><span style="font-size:0.6em;font-weight:normal;opacity:0.85;">'
            f'Upload a new model and set requirements above to see go/no-go evaluation.'
            f'</span>'
            f'</div>'
        )
    return html


# ---------------------------------------------------------------------------
# Customer / PM — charts
# ---------------------------------------------------------------------------

def build_ranked_bar(df, thresholds=None, new_models=None):
    """Horizontal bar chart: one bar per model, peak throughput, colored by status if thresholds set.

    When new_models is provided, status colors apply only to those models; existing models
    are shown in neutral gray for reference.
    """
    if THROUGHPUT_COL not in df.columns:
        return None

    active = {k: v for k, v in (thresholds or {}).items() if v is not None and v > 0 and k in df.columns}

    model_data = []
    for model, grp in df.groupby("Model"):
        peak = grp[THROUGHPUT_COL].max()
        status = None
        # Only evaluate thresholds for newly uploaded models
        if active and (new_models is None or model in new_models):
            pass_cols = []
            for col, thresh in active.items():
                passed = grp[col] <= thresh if col == TTFT_COL else grp[col] >= thresh
                pass_cols.append(passed)
            all_pass = pd.concat(pass_cols, axis=1).all(axis=1)
            pass_pct = all_pass.mean() * 100
            status = "GO" if pass_pct >= 80 else "CAUTION" if pass_pct >= 50 else "NO-GO"
        model_data.append({"model": model, "peak": peak, "status": status})

    model_data.sort(key=lambda x: x["peak"])

    labels = [x["model"] for x in model_data]
    peaks  = [x["peak"]  for x in model_data]

    has_status = any(x["status"] is not None for x in model_data)
    if has_status:
        color_map  = {"GO": "#2d6a2d", "CAUTION": "#c8960c", "NO-GO": "#8b1a1a"}
        bar_colors = [
            color_map[x["status"]] if x["status"] is not None else "#9e9e9e"
            for x in model_data
        ]
    else:
        bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(model_data))]

    fig = go.Figure(go.Bar(
        x=peaks, y=labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{p:,.0f}" for p in peaks],
        textposition="outside",
        textfont=dict(size=11),
    ))

    if has_status:
        for status, color in [("GO", "#2d6a2d"), ("CAUTION", "#c8960c"), ("NO-GO", "#8b1a1a")]:
            fig.add_trace(go.Bar(
                x=[None], y=[None], orientation="h",
                marker_color=color, name=STATUS_EMOJI[status], showlegend=True,
            ))
        if new_models:
            fig.add_trace(go.Bar(
                x=[None], y=[None], orientation="h",
                marker_color="#9e9e9e", name="Existing model (reference)", showlegend=True,
            ))

    fig.update_layout(
        title="Peak Throughput per Model",
        xaxis_title="Peak Throughput (t/s)",
        height=max(320, len(labels) * 44 + 110),
        margin=dict(l=120, r=110, t=60, b=40),
        barmode="overlay",
        legend=dict(title="SLA Status", x=0.6, y=0.1),
        showlegend=has_status,
    )
    return fig


def build_pareto_scatter(df, new_models=None):
    """One dot per model at its (min TTFT, peak throughput) best point.

    Falls back to Min Latency if TTFT column is absent.
    """
    if THROUGHPUT_COL not in df.columns:
        return None

    use_ttft = TTFT_COL in df.columns
    x_col    = TTFT_COL if use_ttft else LATENCY_COL
    if x_col not in df.columns:
        return None

    summary = (
        df.groupby("Model")
        .agg(peak_tp=(THROUGHPUT_COL, "max"), min_x=(x_col, "min"))
        .reset_index()
    )

    if new_models:
        is_new  = summary["Model"].isin(new_models)
        summary = pd.concat([summary[is_new], summary[~is_new]], ignore_index=True)

    x_label = (
        "Min TTFT (ms)  ←  lower is better"
        if use_ttft else
        "Min Latency (ms)  ←  lower is better"
    )
    title = (
        "Performance Frontier — Peak Throughput vs Min TTFT"
        if use_ttft else
        "Performance Frontier — Peak Throughput vs Min Latency"
    )

    fig = px.scatter(
        summary, x="min_x", y="peak_tp",
        color="Model", text="Model",
        color_discrete_sequence=PALETTE,
        labels={"min_x": x_label, "peak_tp": "Peak Throughput (t/s)  ↑  higher is better"},
        title=title,
        height=420,
    )
    fig.update_traces(textposition="top center", marker=dict(size=14))
    fig.update_layout(
        legend_title="Model",
        annotations=[dict(
            x=0.02, y=0.97, xref="paper", yref="paper",
            text="Top-left = best tradeoff",
            showarrow=False, font=dict(size=10, color="gray"),
            xanchor="left", yanchor="top",
        )],
    )
    _emphasise_new_models(fig, new_models)
    return fig


def build_peak_heatmap(df):
    """model × profile grid coloured by peak throughput, columns labelled with workload config."""
    if THROUGHPUT_COL not in df.columns:
        return None

    label_map = _profile_label_map(df)
    pivot     = df.groupby(["Model", "Profile"])[THROUGHPUT_COL].max().reset_index()
    if label_map:
        pivot["Profile"] = pivot["Profile"].map(label_map).fillna(pivot["Profile"])
    hm = pivot.pivot(index="Model", columns="Profile", values=THROUGHPUT_COL)

    fig = px.imshow(
        hm,
        text_auto=".3s",
        color_continuous_scale="Blues",
        title="Peak Throughput (t/s) — Model × Workload Profile",
        labels={"color": "Peak TP (t/s)"},
        aspect="auto",
    )
    fig.update_layout(
        height=520,
        coloraxis_colorbar=dict(title="t/s"),
        xaxis_title="Workload Profile (input tokens → output tokens)",
        annotations=[dict(
            x=0.5, y=-0.18, xref="paper", yref="paper",
            text="Darker = faster  ·  Blank = not tested for this model/profile combination",
            showarrow=False, font=dict(size=10, color="gray"), xanchor="center",
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# Shared highlight helper
# ---------------------------------------------------------------------------

def _emphasise_new_models(fig, new_models):
    """Fade existing-model traces; put new-model traces first in fig.data (= top of legend)."""
    if not new_models:
        return fig
    for trace in fig.data:
        model = trace.name.split(",")[0].strip()
        trace.update(opacity=1.0 if model in new_models else 0.25)
    new_t = [t for t in fig.data if t.name.split(",")[0].strip() in new_models]
    old_t = [t for t in fig.data if t.name.split(",")[0].strip() not in new_models]
    fig.data = tuple(new_t + old_t)
    return fig


# ---------------------------------------------------------------------------
# Engineer — performance overview
# ---------------------------------------------------------------------------

def build_box_plots(df, new_models=None):
    """Throughput distribution per model across all profiles and batch sizes."""
    if THROUGHPUT_COL not in df.columns:
        return None

    # Put new models first on the x-axis
    if new_models:
        new_ms  = sorted(m for m in df["Model"].unique() if m in new_models)
        old_ms  = sorted(m for m in df["Model"].unique() if m not in new_models)
        cat_order = {"Model": new_ms + old_ms}
    else:
        cat_order = {}

    fig = px.box(
        df, x="Model", y=THROUGHPUT_COL,
        color="Model", points="all",
        color_discrete_sequence=PALETTE,
        category_orders=cat_order,
        title="Throughput Distribution per Model (all profiles & batch sizes)",
        labels={THROUGHPUT_COL: "Throughput (t/s)"},
    )
    fig.update_layout(height=460, showlegend=False)
    _emphasise_new_models(fig, new_models)
    return fig


def build_metric_comparison(df, new_models=None):
    """Normalized grouped bar chart: each model's score as % of best performer per metric."""
    COMPARE_METRICS = [
        ("Throughput",     THROUGHPUT_COL,                    "higher"),
        ("TTFT",           TTFT_COL,                          "lower"),
        ("Gen Speed/user", "Gen Speed (t/s/user)",            "higher"),
        ("TP/box",         "Throughput / box (t/s/hardware)", "higher"),
        ("RPM",            RPM_COL,                           "higher"),
    ]

    available = [(lbl, col, d) for lbl, col, d in COMPARE_METRICS if col in df.columns]
    if not available:
        return None

    model_means = df.groupby("Model")[[col for _, col, _ in available]].mean()

    normalized = {}
    for lbl, col, direction in available:
        vals = model_means[col]
        if direction == "higher":
            best = vals.max()
            normalized[lbl] = (vals / best * 100).clip(0, 100) if best > 0 else vals * 0 + 100
        else:
            best = vals.min()
            normalized[lbl] = (best / vals * 100).clip(0, 100) if vals.max() > 0 else vals * 0 + 100

    norm_df = pd.DataFrame(normalized)
    norm_df.index.name = "Model"
    melted  = norm_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score (%)")

    # New models first in legend
    if new_models:
        new_ms = sorted(m for m in melted["Model"].unique() if m in new_models)
        old_ms = sorted(m for m in melted["Model"].unique() if m not in new_models)
        cat_order = {"Model": new_ms + old_ms}
    else:
        cat_order = {}

    fig = px.bar(
        melted, x="Metric", y="Score (%)", color="Model",
        barmode="group",
        color_discrete_sequence=PALETTE,
        category_orders=cat_order,
        title="Relative Performance per Metric (100% = best model in dataset)",
        labels={"Score (%)": "Score (% of best)", "Metric": "Metric"},
        height=460,
    )
    fig.update_layout(
        yaxis_range=[0, 115],
        legend_title="Model",
        yaxis_title="Score (% of best)",
        annotations=[dict(
            x=0.01, y=1.04, xref="paper", yref="paper",
            text="All metrics normalized: 100% = top performer  ·  For TTFT, lower original value = higher score",
            showarrow=False, font=dict(size=9, color="gray"), xanchor="left",
        )],
    )
    _emphasise_new_models(fig, new_models)
    return fig


# ---------------------------------------------------------------------------
# Engineer — scaling analysis
# ---------------------------------------------------------------------------

def compute_scaling_check(df, new_models=None):
    """
    Within-model scaling check: does throughput grow monotonically with batch size?
    Returns (summary_md, efficiency_fig, regression_df).
    """
    if THROUGHPUT_COL not in df.columns or "Batch Size" not in df.columns:
        return "Throughput or Batch Size column not found.", None, pd.DataFrame()

    df = df.copy()
    df["_eff"] = df[THROUGHPUT_COL] / df["Batch Size"]

    # New models first so they get the vivid palette colors and top legend slots
    if new_models:
        is_new    = df["Model"].isin(new_models)
        eff_sorted = pd.concat([df[is_new], df[~is_new]]).sort_values(["Model", "Batch Size"])
    else:
        eff_sorted = df.sort_values("Batch Size")

    eff_fig = px.line(
        eff_sorted, x="Batch Size", y="_eff",
        color="Model", line_dash="Profile",
        title="Throughput Scaling Efficiency (t/s per concurrent request)",
        labels={"_eff": "Throughput / Batch Size", "Batch Size": "Batch Size"},
        color_discrete_sequence=PALETTE,
        height=460,
    )
    eff_fig.update_layout(
        legend_title="Model / Profile",
        yaxis_title="Throughput / Batch Size (t/s per req)",
        annotations=[dict(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text="Flat/slowly-dropping = healthy scaling  |  Steep drop = diminishing returns",
            showarrow=False, font=dict(size=10, color="gray"), xanchor="left",
        )],
    )
    _emphasise_new_models(eff_fig, new_models)

    regressions = []
    for (model, profile), grp in df.groupby(["Model", "Profile"]):
        g = grp.sort_values("Batch Size").reset_index(drop=True)
        for i in range(1, len(g)):
            tp_prev = g.loc[i - 1, THROUGHPUT_COL]
            tp_curr = g.loc[i,     THROUGHPUT_COL]
            bs_prev = g.loc[i - 1, "Batch Size"]
            bs_curr = g.loc[i,     "Batch Size"]
            if tp_curr < tp_prev:
                drop_pct = (tp_prev - tp_curr) / tp_prev * 100
                regressions.append({
                    "Model":            model,
                    "Profile":          profile,
                    "Batch Size":       f"{int(bs_prev)} → {int(bs_curr)}",
                    "Throughput (t/s)": f"{tp_prev:,.0f} → {tp_curr:,.0f}",
                    "Drop":             f"{drop_pct:.1f}%",
                })

    if regressions:
        regression_df = pd.DataFrame(regressions)
    else:
        regression_df = pd.DataFrame([{
            "Model":            "— No regressions detected —",
            "Profile":          "",
            "Batch Size":       "",
            "Throughput (t/s)": "All models scale monotonically with batch size",
            "Drop":             "",
        }])

    n = len(regressions)
    summary_md = (
        f"**{n} throughput regression{'s' if n != 1 else ''} detected** — "
        "cases where adding more concurrent requests *decreases* total throughput."
        if n > 0 else
        "**No throughput regressions detected** — all models scale monotonically with batch size."
    )
    return summary_md, eff_fig, regression_df


# ---------------------------------------------------------------------------
# Engineer — config sensitivity
# ---------------------------------------------------------------------------

def build_engineer_chart(df, sweep_col, new_mp=None):
    has_tp  = THROUGHPUT_COL in df.columns
    has_lat = LATENCY_COL    in df.columns
    if (not has_tp and not has_lat) or sweep_col is None:
        return None

    df = df.copy()
    df["_mp"] = mp_label(df)

    def _label_is_new(label):
        parts = label.split(" / ", 1)
        m = parts[0]
        p = parts[1] if len(parts) > 1 else "—"
        return (m, p) in new_mp

    # New (model, profile) labels first — vivid palette colors and top legend slots
    all_labels = sorted(df["_mp"].unique())
    if new_mp:
        new_labels = [l for l in all_labels if _label_is_new(l)]
        old_labels = [l for l in all_labels if not _label_is_new(l)]
        models = new_labels + old_labels
    else:
        models = all_labels

    def _trace_style(label, idx):
        is_new = new_mp and _label_is_new(label)
        return dict(
            color=PALETTE[idx % len(PALETTE)],
            width=3 if is_new else 1.5,
        ), dict(
            size=8 if is_new else 5,
        ), 1.0 if is_new else 0.25

    if has_tp and has_lat:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[THROUGHPUT_COL, "Max Latency (ms)"],
            shared_xaxes=True, vertical_spacing=0.10,
        )
        for i, label in enumerate(models):
            grp = df[df["_mp"] == label].sort_values(sweep_col)
            line_kw, marker_kw, opacity = _trace_style(label, i)
            fig.add_trace(go.Scatter(
                name=label, x=grp[sweep_col], y=grp[THROUGHPUT_COL],
                mode="lines+markers", line=line_kw, marker=marker_kw,
                opacity=opacity, legendgroup=label,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                name=label, x=grp[sweep_col], y=grp[LATENCY_COL],
                mode="lines+markers", line=line_kw, marker=marker_kw,
                opacity=opacity, legendgroup=label, showlegend=False,
            ), row=2, col=1)
        fig.update_yaxes(title_text="Throughput (t/s)", row=1, col=1)
        fig.update_yaxes(title_text="Latency (ms)",     row=2, col=1)
        fig.update_xaxes(title_text=sweep_col,          row=2, col=1)
    else:
        metric  = THROUGHPUT_COL if has_tp else LATENCY_COL
        y_label = "Throughput (t/s)" if has_tp else "Latency (ms)"
        fig     = go.Figure()
        for i, label in enumerate(models):
            grp = df[df["_mp"] == label].sort_values(sweep_col)
            line_kw, marker_kw, opacity = _trace_style(label, i)
            fig.add_trace(go.Scatter(
                name=label, x=grp[sweep_col], y=grp[metric],
                mode="lines+markers", line=line_kw, marker=marker_kw, opacity=opacity,
            ))
        fig.update_layout(xaxis_title=sweep_col, yaxis_title=y_label)

    fig.update_layout(
        title=f"Config Sensitivity — {sweep_col} Sweep, All Models",
        legend_title="Model / Profile",
        hovermode="x unified",
        height=600,
    )
    return fig


# ---------------------------------------------------------------------------
# Engineer — PCA + t-SNE
# ---------------------------------------------------------------------------

def _compute_tsne(df, new_mp=None):
    """t-SNE 2D scatter coloured by model; newly uploaded (model, profile) shown as gold stars."""
    available = [c for c in PCA_FEATURE_COLS if c in df.columns]
    if len(available) < 3:
        return None

    # Keep rows that have at least half the feature columns non-null,
    # then fill the rest with per-column medians so partial rows are still plotted.
    min_cols = max(3, len(available) // 2)
    extra_cols = ["Model", "Profile"] + (["Batch Size"] if "Batch Size" in df.columns else [])
    sub = df[available + extra_cols].copy()
    sub = sub.dropna(subset=available, thresh=min_cols)
    for col in available:
        sub[col] = sub[col].fillna(sub[col].median())
    if len(sub) < 4:
        return None

    X_scaled = StandardScaler().fit_transform(np.log1p(sub[available].values.clip(min=0)))

    perplexity  = min(30, len(X_scaled) - 1)
    tsne_coords = TSNE(
        n_components=2, random_state=42, perplexity=perplexity,
        max_iter=1000, init="pca",
    ).fit_transform(X_scaled)

    plot_df = sub[["Model", "Profile"]].copy()
    if "Batch Size" in sub.columns:
        plot_df["Batch Size"] = sub["Batch Size"]
    plot_df["tSNE-1"] = tsne_coords[:, 0]
    plot_df["tSNE-2"] = tsne_coords[:, 1]

    hover = {"Batch Size": True} if "Batch Size" in plot_df.columns else {}
    fig = px.scatter(
        plot_df, x="tSNE-1", y="tSNE-2",
        color="Model", symbol="Profile",
        hover_data=hover,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        title="t-SNE — Models clustered by performance similarity",
        height=580,
    )

    new_mp = new_mp or set()
    if new_mp:
        for trace in fig.data:
            parts = trace.name.split(",", 1)
            m = parts[0].strip()
            p = parts[1].strip() if len(parts) > 1 else "—"
            if (m, p) in new_mp:
                trace.marker.size   = 18
                trace.marker.symbol = "star"
                trace.marker.line   = dict(width=2, color="#FFD700")

    annotations = [dict(
        x=0.01, y=0.99, xref="paper", yref="paper",
        text="Points close together = similar performance fingerprint across all metrics",
        showarrow=False, font=dict(size=10, color="gray"), xanchor="left", yanchor="top",
    )]
    if new_mp:
        label = ", ".join(f"{m} {p}" for m, p in sorted(new_mp))
        annotations.append(dict(
            x=0.01, y=0.93, xref="paper", yref="paper",
            text=f"★ Gold star = newly uploaded ({label})",
            showarrow=False, font=dict(size=10, color="#b8860b"), xanchor="left", yanchor="top",
        ))

    fig.update_layout(legend_title="Model / Profile", annotations=annotations)
    return fig


# ---------------------------------------------------------------------------
# Shared analysis pipeline
# ---------------------------------------------------------------------------

def _run_analysis(df, thresh_tp=0, thresh_gen=0, thresh_ttft=0, new_mp=None, run_tsne=False):
    thresholds = {
        THROUGHPUT_COL:           thresh_tp   or 0,
        "Gen Speed (t/s/user)":   thresh_gen  or 0,
        TTFT_COL:                 thresh_ttft or 0,
    }
    sweep_col = detect_sweep_col(df)

    # Derive model-level names for charts that work per-model (not per profile)
    new_models = {m for m, p in new_mp} if new_mp else None

    # Go/no-go evaluation is scoped to newly uploaded models only.
    # When no new models are present, suppress threshold coloring so the banner
    # shows neutral headroom/throughput info instead of a go/no-go verdict.
    if new_mp:
        gonogo_df    = df[df["Model"].isin(new_models)]
        ranking_df   = compute_ranking(gonogo_df, thresholds)
        summary_html = build_summary_html(ranking_df, thresholds)
        ranked_bar   = build_ranked_bar(df, thresholds, new_models=new_models)
    else:
        ranking_df   = compute_ranking(df)
        summary_html = build_summary_html(ranking_df, {})
        ranked_bar   = build_ranked_bar(df, {})

    pareto_scatter    = build_pareto_scatter(df, new_models=new_models)
    peak_heatmap      = build_peak_heatmap(df)

    box_plots         = build_box_plots(df, new_models=new_models)
    metric_comparison = build_metric_comparison(df, new_models=new_models)

    scaling_md, scaling_chart, regression_df = compute_scaling_check(df, new_models=new_models)
    eng_chart  = build_engineer_chart(df.copy(), sweep_col, new_mp=new_mp)
    tsne_fig   = _compute_tsne(df, new_mp) if run_tsne else None

    return (
        summary_html, ranked_bar, pareto_scatter,
        box_plots, metric_comparison,
        scaling_md, scaling_chart, regression_df,
        eng_chart, tsne_fig, peak_heatmap,
    )


def perform_analysis(files, thresh_tp=0, thresh_gen=0, thresh_ttft=0):
    base = _BASE_DF
    if not files:
        if base is None:
            msg = _BASE_DF_ERR or "No dataset files found in `dataset/`."
            return None, None, "", msg, None, None, None, None, msg, None, None, None, None, None
        return (base, None, "") + tuple(_run_analysis(base, thresh_tp, thresh_gen, thresh_ttft, run_tsne=True))
    try:
        uploaded_df, errors, warnings = read_files(files)
    except Exception as e:
        err_html = _build_validation_html([f"Error reading uploaded files: {e}"], [])
        fail_msg = "Upload failed — see error above."
        return None, None, err_html, fail_msg, None, None, None, None, fail_msg, None, None, None, None, None
    validation_html = _build_validation_html(errors, warnings)
    if uploaded_df is None or uploaded_df.empty:
        no_data_msg = (
            "No usable data found in the uploaded files — see the errors above."
            if errors else "No data found in uploaded files."
        )
        return None, None, validation_html, no_data_msg, None, None, None, None, no_data_msg, None, None, None, None, None
    new_mp = set(zip(uploaded_df["Model"], uploaded_df["Profile"]))
    if base is not None:
        combined = pd.concat([base, uploaded_df], ignore_index=True)
        key_cols = [c for c in ["Model", "Profile", "Batch Size"] if c in combined.columns]
        if key_cols:
            combined = combined.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    else:
        combined = uploaded_df
    return (combined, new_mp, validation_html) + tuple(
        _run_analysis(combined, thresh_tp, thresh_gen, thresh_ttft, new_mp, run_tsne=True)
    )


def apply_thresholds(df, new_mp, thresh_tp=0, thresh_gen=0, thresh_ttft=0):
    if df is None:
        msg = "No data available. Check that dataset files exist in `dataset/`."
        return msg, None, None, None, None, msg, None, None, None, None, None
    return _run_analysis(df, thresh_tp, thresh_gen, thresh_ttft, new_mp or None)


def reset_highlights(df, thresh_tp=0, thresh_gen=0, thresh_ttft=0):
    """Re-render all charts with no model highlighted; clear new_models_state and validation banner."""
    if df is None:
        msg = "No data available. Check that dataset files exist in `dataset/`."
        return None, "", msg, None, None, None, None, msg, None, None, None, None, None
    return (None, "") + tuple(_run_analysis(df, thresh_tp, thresh_gen, thresh_ttft, new_models=None, run_tsne=True))


# Pre-compute the base analysis so every tab has content from the first page load.
_INIT = _run_analysis(_BASE_DF, run_tsne=True) if _BASE_DF is not None else (None,) * 11

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Performance Analyzer") as app:
    gr.Markdown(
        "# Performance Analyzer\n"
        "Models A–K are pre-loaded and shown below. "
        "To add more models, drop their `.xlsx` sweep files in the upload area and click **Add Models to Analysis**."
    )

    with gr.Row():
        upload = gr.File(
            label="Additional Models (.xlsx) — drag-and-drop to add to the analysis (optional)",
            file_types=[".xlsx"],
            file_count="multiple",
        )
    with gr.Row():
        analyze_btn = gr.Button("Add Models to Analysis", variant="primary",   size="lg")
        reset_btn   = gr.Button("Show All Models Equally", variant="secondary", size="lg")

    validation_out = gr.Markdown(value="", sanitize_html=False)

    with gr.Tabs():

        # ── Tab 1: Customer / PM ─────────────────────────────────────────────
        with gr.Tab("Customer / PM"):
            with gr.Accordion("Your Performance Requirements (go/no-go thresholds)", open=True):
                gr.Markdown(
                    "Set one or more thresholds below, then upload a new model — "
                    "go/no-go is evaluated **only for newly uploaded models**. "
                    "Pre-loaded models A–K are shown as gray reference bars.\n\n"
                    "A model is **GO** if it meets all active requirements in ≥ 80% of tested "
                    "configurations, **CAUTION** in 50–79%, **NO-GO** below 50%. "
                    "Throughput and Gen Speed default to **0 (disabled)**; "
                    "TTFT defaults to the dataset maximum (all rows pass — effectively disabled)."
                )
                with gr.Row():
                    _tp   = _SLIDER_STATS.get(THROUGHPUT_COL, {})
                    _gen  = _SLIDER_STATS.get("Gen Speed (t/s/user)", {})
                    _ttft = _SLIDER_STATS.get(TTFT_COL, {})

                    # Throughput / gen: minimum=0 so the slider can reach the
                    # disabled state (0 = criterion skipped).  Maximum = strongest
                    # model's median, i.e. the most demanding sensible requirement.
                    _tp_max   = int(_tp.get("model_max_median",  2_000_000))
                    _gen_max  = int(_gen.get("model_max_median",      1200))

                    # TTFT: minimum = best (lowest) model median so the slider
                    # doesn't ask for physically impossible latency.
                    # Maximum = actual highest individual TTFT row in the dataset —
                    # at this position every row passes (criterion effectively off).
                    _ttft_min = round(_ttft.get("model_min_median",     0), 1)
                    _ttft_max = round(_ttft.get("max",                 40), 1)

                    thresh_tp = gr.Slider(
                        label="Min Throughput (t/s)",
                        minimum=0, maximum=_tp_max,
                        step=max(1000, _tp_max // 40),
                        value=0,
                        info="Minimum throughput the model must sustain. "
                             "0 = criterion disabled. "
                             "Right = strongest model's median (most strict).",
                    )
                    thresh_gen = gr.Slider(
                        label="Min Gen Speed (t/s/user)",
                        minimum=0, maximum=_gen_max,
                        step=max(1, _gen_max // 40),
                        value=0,
                        info="Minimum per-user token generation speed. "
                             "0 = criterion disabled. "
                             "Right = fastest model's median (most strict).",
                    )
                    thresh_ttft = gr.Slider(
                        label="Max TTFT (ms)",
                        minimum=_ttft_min, maximum=_ttft_max,
                        step=max(0.5, round((_ttft_max - _ttft_min) / 40, 1)),
                        value=_ttft_max,
                        info="Maximum time-to-first-token allowed. "
                             "Right = highest observed TTFT in dataset (all rows pass). "
                             "Left = lowest model median (most strict).",
                    )
                update_btn = gr.Button("Update Go/No-Go", variant="secondary", size="sm")

            summary_output = gr.Markdown(value=_INIT[0], sanitize_html=False)
            with gr.Row():
                ranked_bar_out = gr.Plot(label="Ranked by Peak Throughput",      value=_INIT[1])
                pareto_out     = gr.Plot(label="Performance Frontier",            value=_INIT[2])
            peak_heatmap_out   = gr.Plot(label="Peak Throughput by Model × Workload Profile", value=_INIT[10])

        # ── Tab 2: Engineer ──────────────────────────────────────────────────
        with gr.Tab("Engineer"):

            with gr.Accordion("Performance Overview", open=True):
                with gr.Row():
                    box_plots_out   = gr.Plot(label="Throughput Distribution per Model",  value=_INIT[3])
                    metric_comp_out = gr.Plot(label="Relative Performance Comparison",    value=_INIT[4])

            with gr.Accordion("Scaling Analysis", open=True):
                scaling_md_out    = gr.Markdown(value=_INIT[5])
                scaling_chart_out = gr.Plot(label="Throughput Scaling Efficiency",        value=_INIT[6])
                regression_out    = gr.DataFrame(label="Throughput Regressions",          value=_INIT[7])

            with gr.Accordion("Config Sensitivity", open=True):
                eng_chart_out = gr.Plot(label="Throughput & Latency vs Batch Size — All Models", value=_INIT[8])

            tsne_out = gr.Plot(label="Model Similarity (t-SNE)", value=_INIT[9])

    df_state         = gr.State(value=_BASE_DF)
    new_models_state = gr.State(value=None)

    _analysis_outputs = [
        summary_output, ranked_bar_out, pareto_out,
        box_plots_out, metric_comp_out,
        scaling_md_out, scaling_chart_out, regression_out,
        eng_chart_out, tsne_out, peak_heatmap_out,
    ]

    analyze_btn.click(
        fn=perform_analysis,
        inputs=[upload, thresh_tp, thresh_gen, thresh_ttft],
        outputs=[df_state, new_models_state, validation_out] + _analysis_outputs,
    )
    update_btn.click(
        fn=apply_thresholds,
        inputs=[df_state, new_models_state, thresh_tp, thresh_gen, thresh_ttft],
        outputs=_analysis_outputs,
    )
    reset_btn.click(
        fn=reset_highlights,
        inputs=[df_state, thresh_tp, thresh_gen, thresh_ttft],
        outputs=[new_models_state, validation_out] + _analysis_outputs,
    )

app.launch(theme=gr.themes.Soft())
