# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding="utf-8")
"""
Qwen 2.5 — Regression & Error Metrics
Calculates RMSE, R², MSE, MAE, MAPE, Adjusted-R², Pearson-r for:
  Part A: All 19 LLMs  (MMLU & HumanEval scaling laws)
  Part B: Qwen2.5 only (all 12 evaluation metrics × each model vs GPT-4 baseline)
  Part C: Per-model error summary vs GPT-4 reference
Saves results to ../evaluation/qwen25_evaluation/
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

OUT_DIR = "../evaluation/qwen25_evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def mse(y_true, y_pred):
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def r2_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

def adj_r2(r2, n, k=1):
    return float(1 - (1 - r2) * (n - 1) / (n - k - 1 + 1e-12))

def log_linear_fit(x_vals, y_vals):
    """Fit y ~ a*log(x) + b, return (y_pred, slope, intercept, r2, adj_r2, rmse_, mse_, mae_, mape_, pearson_r)."""
    lx = np.log(np.array(x_vals, dtype=float))
    y  = np.array(y_vals,  dtype=float)
    slope, intercept, r, p, se = stats.linregress(lx, y)
    y_pred = slope * lx + intercept
    r2  = r2_score(y, y_pred)
    ar2 = adj_r2(r2, len(y))
    return y_pred, slope, intercept, r2, ar2, rmse(y, y_pred), mse(y, y_pred), mae(y, y_pred), mape(y, y_pred), r

PALETTE = ["#0ea5e9","#6366f1","#10b981","#f59e0b","#ef4444","#8b5cf6"]
AMBER   = "#f59e0b"
AMBER_D = "#b45309"

# ═══════════════════════════════════════════════════════════════════════════════
# PART A — ALL 19 LLMs: MMLU & HumanEval Scaling Law Regression
# ═══════════════════════════════════════════════════════════════════════════════
ALL_MODELS = [
    {"name":"SmolLM2-135M",   "family":"SmolLM2",   "params_B":0.135,"mmlu":30.1,"humaneval":5.4},
    {"name":"Qwen2.5-0.5B",   "family":"Qwen2.5",   "params_B":0.5,  "mmlu":45.4,"humaneval":28.9},
    {"name":"TinyLlama-1.1B", "family":"TinyLlama", "params_B":1.1,  "mmlu":25.6,"humaneval":8.0},
    {"name":"Llama-3.2-1B",   "family":"Llama 3.x", "params_B":1.0,  "mmlu":44.7,"humaneval":25.0},
    {"name":"SmolLM2-1.7B",   "family":"SmolLM2",   "params_B":1.7,  "mmlu":48.3,"humaneval":21.0},
    {"name":"Qwen2.5-1.5B",   "family":"Qwen2.5",   "params_B":1.5,  "mmlu":60.9,"humaneval":37.2},
    {"name":"Gemma-2-2B",     "family":"Gemma",      "params_B":2.0,  "mmlu":52.2,"humaneval":36.0},
    {"name":"Qwen2.5-3B",     "family":"Qwen2.5",   "params_B":3.0,  "mmlu":65.6,"humaneval":55.5},
    {"name":"Llama-3.2-3B",   "family":"Llama 3.x", "params_B":3.0,  "mmlu":58.0,"humaneval":40.0},
    {"name":"Phi-3.5-mini",   "family":"Phi",        "params_B":3.8,  "mmlu":69.0,"humaneval":59.0},
    {"name":"Mistral-7B",     "family":"Mistral",    "params_B":7.0,  "mmlu":63.0,"humaneval":41.0},
    {"name":"Qwen2.5-7B",     "family":"Qwen2.5",   "params_B":7.0,  "mmlu":74.2,"humaneval":72.0},
    {"name":"Llama-3.1-8B",   "family":"Llama 3.x", "params_B":8.0,  "mmlu":68.4,"humaneval":62.0},
    {"name":"Gemma-2-9B",     "family":"Gemma",      "params_B":9.0,  "mmlu":71.3,"humaneval":55.0},
    {"name":"Phi-3-medium",   "family":"Phi",        "params_B":14.0, "mmlu":78.0,"humaneval":65.0},
    {"name":"Qwen2.5-14B",    "family":"Qwen2.5",   "params_B":14.0, "mmlu":79.7,"humaneval":78.0},
    {"name":"Mixtral-8x7B",   "family":"Mistral",    "params_B":46.7, "mmlu":71.0,"humaneval":40.2},
    {"name":"Llama-3.1-70B",  "family":"Llama 3.x", "params_B":70.0, "mmlu":83.6,"humaneval":80.0},
    {"name":"Qwen2.5-72B",    "family":"Qwen2.5",   "params_B":72.0, "mmlu":86.1,"humaneval":86.0},
]

FAMILY_COLORS = {"Qwen2.5":"#f59e0b","Llama 3.x":"#6366f1","Phi":"#10b981",
                 "Gemma":"#ec4899","Mistral":"#8b5cf6","TinyLlama":"#64748b","SmolLM2":"#94a3b8"}

params_all = [m["params_B"]    for m in ALL_MODELS]
mmlu_all   = [m["mmlu"]        for m in ALL_MODELS]
heval_all  = [m["humaneval"]   for m in ALL_MODELS]
names_all  = [m["name"]        for m in ALL_MODELS]
fam_all    = [m["family"]      for m in ALL_MODELS]

mmlu_pred,  s_m, i_m, r2_m, ar2_m, rmse_m, mse_m, mae_m, mape_m, pr_m  = log_linear_fit(params_all, mmlu_all)
heval_pred, s_h, i_h, r2_h, ar2_h, rmse_h, mse_h, mae_h, mape_h, pr_h  = log_linear_fit(params_all, heval_all)

# Per-model residuals
mmlu_resid  = [float(a - p) for a, p in zip(mmlu_all,  mmlu_pred)]
heval_resid = [float(a - p) for a, p in zip(heval_all, heval_pred)]

part_a_results = {
    "MMLU_scaling_law": {
        "formula": "MMLU = {:.3f}*ln(params_B) + {:.3f}".format(s_m, i_m),
        "R2":            round(r2_m,   4),
        "Adjusted_R2":   round(ar2_m,  4),
        "RMSE":          round(rmse_m, 4),
        "MSE":           round(mse_m,  4),
        "MAE":           round(mae_m,  4),
        "MAPE_pct":      round(mape_m, 4),
        "Pearson_r":     round(pr_m,   4),
        "per_model": {m["name"]: {"actual": m["mmlu"], "predicted": round(float(p), 2),
                                   "residual": round(float(r), 2)}
                      for m, p, r in zip(ALL_MODELS, mmlu_pred, mmlu_resid)},
    },
    "HumanEval_scaling_law": {
        "formula": "HumanEval = {:.3f}*ln(params_B) + {:.3f}".format(s_h, i_h),
        "R2":            round(r2_h,   4),
        "Adjusted_R2":   round(ar2_h,  4),
        "RMSE":          round(rmse_h, 4),
        "MSE":           round(mse_h,  4),
        "MAE":           round(mae_h,  4),
        "MAPE_pct":      round(mape_h, 4),
        "Pearson_r":     round(pr_h,   4),
        "per_model": {m["name"]: {"actual": m["humaneval"], "predicted": round(float(p), 2),
                                   "residual": round(float(r), 2)}
                      for m, p, r in zip(ALL_MODELS, heval_pred, heval_resid)},
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# PART B — Qwen2.5 only: all 12 metrics scaling law
# ═══════════════════════════════════════════════════════════════════════════════
QWEN_MODELS = [
    {"name":"Qwen2.5-0.5B", "params_B":0.5},
    {"name":"Qwen2.5-1.5B", "params_B":1.5},
    {"name":"Qwen2.5-3B",   "params_B":3.0},
    {"name":"Qwen2.5-7B",   "params_B":7.0},
    {"name":"Qwen2.5-14B",  "params_B":14.0},
    {"name":"Qwen2.5-72B",  "params_B":72.0},
]
QWEN_NAMES = [m["name"] for m in QWEN_MODELS]
QWEN_PARAMS = [m["params_B"] for m in QWEN_MODELS]

METRICS_DATA = {
    "intent_accuracy":           {"values":[52.3,63.1,71.4,79.6,84.2,90.1], "lower_is_better":False},
    "code_generation_accuracy":  {"values":[28.9,37.2,55.5,72.0,78.0,86.0], "lower_is_better":False},
    "functional_correctness":    {"values":[31.4,41.8,59.2,75.3,81.6,88.5], "lower_is_better":False},
    "execution_success_rate":    {"values":[52.1,61.3,72.4,84.7,89.2,94.1], "lower_is_better":False},
    "answer_accuracy":           {"values":[45.4,60.9,65.6,74.2,79.7,86.1], "lower_is_better":False},
    "exact_match":               {"values":[31.2,42.8,48.6,58.4,64.1,72.3], "lower_is_better":False},
    "semantic_similarity":       {"values":[68.2,73.6,77.1,82.4,86.3,91.0], "lower_is_better":False},
    "hallucination_rate":        {"values":[28.4,22.1,18.3,13.6,10.2, 7.4], "lower_is_better":True},
    "robustness":                {"values":[48.3,57.9,63.2,71.4,76.8,83.5], "lower_is_better":False},
    "response_time_sec":         {"values":[1.43, 3.33, 6.25,14.29,28.57,100.0],"lower_is_better":True},
    "consistency":               {"values":[61.4,68.9,74.2,80.6,84.7,88.3], "lower_is_better":False},
    "scalability":               {"values":[51.2,58.4,63.7,78.9,82.6,87.4], "lower_is_better":False},
}

# GPT-4 class reference baseline (industry standard)
GPT4_REFERENCE = {
    "intent_accuracy":          90.0,
    "code_generation_accuracy": 87.0,
    "functional_correctness":   82.0,
    "execution_success_rate":   95.0,
    "answer_accuracy":          87.0,
    "exact_match":              59.0,
    "semantic_similarity":      89.0,
    "hallucination_rate":        9.0,
    "robustness":               86.0,
    "response_time_sec":         5.0,
    "consistency":              92.0,
    "scalability":              89.0,
}

part_b_results = {}
for key, meta in METRICS_DATA.items():
    vals = meta["values"]
    y_pred_fit, slope, intercept, r2, ar2, rmse_, mse_, mae_, mape_, pr = log_linear_fit(QWEN_PARAMS, vals)
    ref = GPT4_REFERENCE[key]
    per_model = {}
    for name, p, v, pred in zip(QWEN_NAMES, QWEN_PARAMS, vals, y_pred_fit):
        gap = v - ref if not meta["lower_is_better"] else ref - v
        per_model[name] = {
            "actual":       v,
            "predicted_fit": round(float(pred), 3),
            "residual":     round(float(v - pred), 3),
            "gpt4_ref":     ref,
            "gap_from_gpt4": round(float(gap), 2),
            "abs_error_from_gpt4": round(abs(float(gap)), 2),
        }
    part_b_results[key] = {
        "formula":       f"score = {slope:.3f}*ln(params_B) + {intercept:.3f}",
        "R2":            round(r2,    4),
        "Adjusted_R2":   round(ar2,   4),
        "RMSE":          round(rmse_, 4),
        "MSE":           round(mse_,  4),
        "MAE":           round(mae_,  4),
        "MAPE_pct":      round(mape_, 4),
        "Pearson_r":     round(pr,    4),
        "lower_is_better": meta["lower_is_better"],
        "gpt4_reference": ref,
        "per_model":     per_model,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# PART C — Per-model RMSE / MAE vs GPT-4 baseline (across all 12 metrics)
# ═══════════════════════════════════════════════════════════════════════════════
part_c_results = {}
for i, name in enumerate(QWEN_NAMES):
    actual_normalised = []
    ref_normalised    = []
    for key, meta in METRICS_DATA.items():
        v   = meta["values"][i]
        ref = GPT4_REFERENCE[key]
        # Flip lower-is-better so higher always means better
        if meta["lower_is_better"]:
            if key == "response_time_sec":
                max_v = max(meta["values"]) + 1
                v_n   = 100 * (max_v - v)   / max_v
                r_n   = 100 * (max_v - ref) / max_v
            else:
                v_n = 100 - v
                r_n = 100 - ref
        else:
            v_n = v
            r_n = ref
        actual_normalised.append(v_n)
        ref_normalised.append(r_n)
    part_c_results[name] = {
        "params_B":             QWEN_MODELS[i]["params_B"],
        "RMSE_vs_GPT4":        round(rmse(ref_normalised, actual_normalised), 4),
        "MSE_vs_GPT4":         round(mse( ref_normalised, actual_normalised), 4),
        "MAE_vs_GPT4":         round(mae( ref_normalised, actual_normalised), 4),
        "MAPE_vs_GPT4_pct":    round(mape(ref_normalised, actual_normalised), 4),
        "composite_score":     round(float(np.mean(actual_normalised)), 2),
        "gpt4_composite":      round(float(np.mean(ref_normalised)), 2),
        "gap_from_gpt4":       round(float(np.mean(actual_normalised)) - float(np.mean(ref_normalised)), 2),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE JSON
# ═══════════════════════════════════════════════════════════════════════════════
full_results = {
    "title": "Qwen 2.5 — Regression & Error Metrics (RMSE, R², MSE, MAE, MAPE, Pearson-r)",
    "date":  "2026-06-09",
    "metric_definitions": {
        "RMSE": "Root Mean Squared Error — penalises large deviations more than MAE",
        "MSE":  "Mean Squared Error — average squared deviation (RMSE = √MSE)",
        "MAE":  "Mean Absolute Error — average absolute deviation, robust to outliers",
        "MAPE": "Mean Absolute Percentage Error — relative error as percentage",
        "R2":   "Coefficient of Determination — proportion of variance explained by the log-linear model (1=perfect, 0=no better than mean)",
        "Adj_R2": "Adjusted R² — penalises for number of predictors; more honest than R² for small samples",
        "Pearson_r": "Linear correlation coefficient between ln(params) and metric score (-1 to +1)",
        "gpt4_reference": "GPT-4 class score used as industry benchmark baseline",
    },
    "part_a_all_19_llms_scaling":   part_a_results,
    "part_b_qwen25_per_metric":     part_b_results,
    "part_c_per_model_vs_gpt4":     part_c_results,
}
json_path = os.path.join(OUT_DIR, "qwen25_regression_metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(full_results, f, indent=2, ensure_ascii=False)
print(f"JSON saved: {json_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — All-19 LLM Scaling Law (MMLU & HumanEval) + Residuals
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── 1a: MMLU scatter + fit line ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
log_range = np.linspace(np.log(min(params_all)), np.log(max(params_all)), 200)
fit_mmlu  = s_m * log_range + i_m
ax.plot(np.exp(log_range), fit_mmlu, "--", color="#ef4444", linewidth=2,
        label=f"Log-linear fit\nR²={r2_m:.3f}  RMSE={rmse_m:.2f}")
for m, p, r in zip(ALL_MODELS, params_all, mmlu_resid):
    fc = FAMILY_COLORS[m["family"]]
    brd = AMBER_D if m["family"] == "Qwen2.5" else "white"
    lw  = 2.0    if m["family"] == "Qwen2.5" else 0.6
    ax.scatter(p, m["mmlu"], color=fc, edgecolors=brd, linewidths=lw, s=120, zorder=3)
    ax.annotate(m["name"].replace("Qwen2.5-","Q-").replace("Llama-3.","L3."),
                (p, m["mmlu"]), xytext=(4,3), textcoords="offset points", fontsize=6.5,
                fontweight="bold" if m["family"]=="Qwen2.5" else "normal")
ax.set_xscale("log")
ax.set_xlabel("Model Size (B params, log scale)", fontsize=10)
ax.set_ylabel("MMLU Score (%)", fontsize=10)
ax.set_title(f"MMLU Scaling Law — All 19 LLMs\nR²={r2_m:.4f}  Adj-R²={ar2_m:.4f}  "
             f"RMSE={rmse_m:.3f}  MAE={mae_m:.3f}  Pearson r={pr_m:.4f}",
             fontsize=9.5, fontweight="bold")
family_patches = [mpatches.Patch(color=c, label=f) for f,c in FAMILY_COLORS.items()]
ax.legend(handles=family_patches+[plt.Line2D([0],[0],ls="--",color="#ef4444",label=f"Log-linear fit R²={r2_m:.3f}")],
          fontsize=7, loc="upper left")
ax.grid(linestyle="--", alpha=0.3)

# ── 1b: HumanEval scatter + fit line ─────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
fit_heval = s_h * log_range + i_h
ax.plot(np.exp(log_range), fit_heval, "--", color="#ef4444", linewidth=2,
        label=f"Log-linear fit\nR²={r2_h:.3f}  RMSE={rmse_h:.2f}")
for m, p, r in zip(ALL_MODELS, params_all, heval_resid):
    fc = FAMILY_COLORS[m["family"]]
    brd = AMBER_D if m["family"] == "Qwen2.5" else "white"
    lw  = 2.0    if m["family"] == "Qwen2.5" else 0.6
    ax.scatter(p, m["humaneval"], color=fc, edgecolors=brd, linewidths=lw, s=120, zorder=3)
    ax.annotate(m["name"].replace("Qwen2.5-","Q-").replace("Llama-3.","L3."),
                (p, m["humaneval"]), xytext=(4,3), textcoords="offset points", fontsize=6.5,
                fontweight="bold" if m["family"]=="Qwen2.5" else "normal")
ax.set_xscale("log")
ax.set_xlabel("Model Size (B params, log scale)", fontsize=10)
ax.set_ylabel("HumanEval Score (%)", fontsize=10)
ax.set_title(f"HumanEval Scaling Law — All 19 LLMs\nR²={r2_h:.4f}  Adj-R²={ar2_h:.4f}  "
             f"RMSE={rmse_h:.3f}  MAE={mae_h:.3f}  Pearson r={pr_h:.4f}",
             fontsize=9.5, fontweight="bold")
ax.legend(handles=family_patches+[plt.Line2D([0],[0],ls="--",color="#ef4444",label=f"Log-linear fit R²={r2_h:.3f}")],
          fontsize=7, loc="upper left")
ax.grid(linestyle="--", alpha=0.3)

# ── 1c: MMLU Residuals per model ──────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
colors_bar = [FAMILY_COLORS[f] for f in fam_all]
bars = ax.bar(range(len(names_all)), mmlu_resid, color=colors_bar, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="#1e293b", linewidth=1.2)
for xi, (r, name) in enumerate(zip(mmlu_resid, names_all)):
    ax.text(xi, r + (0.5 if r >= 0 else -1.2), f"{r:.1f}",
            ha="center", fontsize=6.5, fontweight="bold",
            color="#166534" if r >= 0 else "#991b1b")
ax.set_xticks(range(len(names_all)))
ax.set_xticklabels([n.replace("Qwen2.5-","Q").replace("Llama-3.","L3.").replace("-","") for n in names_all],
                    rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Residual (Actual − Predicted)", fontsize=10)
ax.set_title(f"MMLU Residuals — Deviation from Log-linear Fit\nRMSE={rmse_m:.3f}  "
             f"MAE={mae_m:.3f}  MAPE={mape_m:.2f}%",
             fontsize=9.5, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ── 1d: HumanEval Residuals per model ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
bars = ax.bar(range(len(names_all)), heval_resid, color=colors_bar, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="#1e293b", linewidth=1.2)
for xi, (r, name) in enumerate(zip(heval_resid, names_all)):
    ax.text(xi, r + (0.5 if r >= 0 else -1.5), f"{r:.1f}",
            ha="center", fontsize=6.5, fontweight="bold",
            color="#166534" if r >= 0 else "#991b1b")
ax.set_xticks(range(len(names_all)))
ax.set_xticklabels([n.replace("Qwen2.5-","Q").replace("Llama-3.","L3.").replace("-","") for n in names_all],
                    rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Residual (Actual − Predicted)", fontsize=10)
ax.set_title(f"HumanEval Residuals — Deviation from Log-linear Fit\nRMSE={rmse_h:.3f}  "
             f"MAE={mae_h:.3f}  MAPE={mape_h:.2f}%",
             fontsize=9.5, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.suptitle("All 19 LLMs — Scaling Law Regression & Residual Analysis",
             fontsize=14, fontweight="bold", y=1.01)
out = os.path.join(OUT_DIR, "regression_all19_scaling.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"Plot saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Qwen2.5: R² and RMSE for each of 12 metrics
# ═══════════════════════════════════════════════════════════════════════════════
metric_keys   = list(METRICS_DATA.keys())
metric_labels = [k.replace("_"," ").title().replace("Pct","%") for k in metric_keys]
r2_vals   = [part_b_results[k]["R2"]   for k in metric_keys]
rmse_vals = [part_b_results[k]["RMSE"] for k in metric_keys]
mae_vals  = [part_b_results[k]["MAE"]  for k in metric_keys]
pr_vals   = [part_b_results[k]["Pearson_r"] for k in metric_keys]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Qwen 2.5 — Scaling Law Fit Quality Across All 12 Metrics",
             fontsize=14, fontweight="bold", y=1.01)

# R²
ax = axes[0, 0]
colors_r2 = ["#10b981" if v >= 0.95 else "#f59e0b" if v >= 0.85 else "#ef4444" for v in r2_vals]
bars = ax.barh(metric_labels, r2_vals, color=colors_r2, edgecolor="white", linewidth=0.5)
for bar, v in zip(bars, r2_vals):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold")
ax.axvline(0.95, color="#10b981", linestyle="--", linewidth=1.2, alpha=0.7, label="R²=0.95 (excellent)")
ax.axvline(0.85, color="#f59e0b", linestyle=":", linewidth=1.2, alpha=0.7, label="R²=0.85 (good)")
ax.set_xlim(0, 1.08); ax.set_xlabel("R² Score", fontsize=11)
ax.set_title("R² (Coefficient of Determination)\nHow well log(params) predicts each metric",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="x", linestyle="--", alpha=0.3)

# RMSE
ax = axes[0, 1]
colors_rmse = ["#10b981" if v < 2 else "#f59e0b" if v < 5 else "#ef4444" for v in rmse_vals]
bars = ax.barh(metric_labels, rmse_vals, color=colors_rmse, edgecolor="white")
for bar, v in zip(bars, rmse_vals):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")
ax.set_xlabel("RMSE (percentage points)", fontsize=11)
ax.set_title("RMSE — Root Mean Squared Error\nDeviation of actual scores from log-linear fit",
             fontsize=10, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.3)

# MAE
ax = axes[1, 0]
colors_mae = ["#10b981" if v < 2 else "#f59e0b" if v < 4 else "#ef4444" for v in mae_vals]
bars = ax.barh(metric_labels, mae_vals, color=colors_mae, edgecolor="white")
for bar, v in zip(bars, mae_vals):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")
ax.set_xlabel("MAE (percentage points)", fontsize=11)
ax.set_title("MAE — Mean Absolute Error\nAverage absolute deviation from fit (outlier-robust)",
             fontsize=10, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.3)

# Pearson r
ax = axes[1, 1]
colors_pr = ["#10b981" if abs(v) >= 0.97 else "#f59e0b" if abs(v) >= 0.90 else "#ef4444" for v in pr_vals]
bars = ax.barh(metric_labels, pr_vals, color=colors_pr, edgecolor="white")
for bar, v in zip(bars, pr_vals):
    xpos = v + 0.005 if v >= 0 else v - 0.005
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5, fontweight="bold")
ax.axvline(0.97,  color="#10b981", linestyle="--", linewidth=1.2, alpha=0.7, label="|r|=0.97 (excellent)")
ax.axvline(-0.97, color="#10b981", linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_xlim(-1.1, 1.1); ax.set_xlabel("Pearson r", fontsize=11)
ax.set_title("Pearson Correlation — ln(params) vs metric\n+1=perfect positive, -1=perfect negative",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
out = os.path.join(OUT_DIR, "regression_qwen25_fit_quality.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"Plot saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Qwen2.5: 12 scaling law fit lines (actual vs predicted)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 3, figsize=(20, 22))
fig.suptitle("Qwen 2.5 — Log-Linear Scaling Fit: Actual vs Predicted (all 12 metrics)",
             fontsize=14, fontweight="bold", y=1.005)
log_q = np.linspace(np.log(0.4), np.log(80), 200)
params_q = np.exp(log_q)

for ax, key in zip(axes.flat, metric_keys):
    meta  = METRICS_DATA[key]
    res   = part_b_results[key]
    vals  = meta["values"]
    slope    = float(res["formula"].split("*")[0].split("= ")[1])
    intercept= float(res["formula"].split("+ ")[1])
    fit_line = slope * log_q + intercept
    # Actual points
    ax.scatter(QWEN_PARAMS, vals, color=AMBER, edgecolors=AMBER_D, s=100, zorder=5, linewidths=1.5)
    for p, v, name in zip(QWEN_PARAMS, vals, QWEN_NAMES):
        ax.annotate(name.replace("Qwen2.5-",""), (p, v), xytext=(4,3),
                    textcoords="offset points", fontsize=7.5, fontweight="bold", color=AMBER_D)
    # Fit line
    ax.plot(params_q, fit_line, "--", color="#6366f1", linewidth=2, label="Log-linear fit")
    # Residual bars
    y_pred_pts = slope * np.log(np.array(QWEN_PARAMS)) + intercept
    for p, v, yp in zip(QWEN_PARAMS, vals, y_pred_pts):
        ax.plot([p, p], [yp, v], color="#ef4444", linewidth=1.5, alpha=0.7)
    # GPT-4 reference
    gpt4_ref = res["gpt4_reference"]
    ax.axhline(gpt4_ref, color="#10b981", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.text(params_q[-1]*0.85, gpt4_ref+0.3, f"GPT-4 ref={gpt4_ref}", fontsize=7, color="#10b981")
    ax.set_xscale("log")
    lib = " ▼" if meta["lower_is_better"] else ""
    ax.set_title(f"{key.replace('_',' ').title()}{lib}\n"
                 f"R²={res['R2']:.4f}  RMSE={res['RMSE']:.3f}  MAE={res['MAE']:.3f}  r={res['Pearson_r']:.4f}",
                 fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Params (B, log)", fontsize=8)
    ax.set_ylabel("Score", fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(fontsize=7)

plt.tight_layout()
out = os.path.join(OUT_DIR, "regression_qwen25_fit_lines.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"Plot saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Per-model RMSE vs GPT-4 baseline
# ═══════════════════════════════════════════════════════════════════════════════
model_names_c = list(part_c_results.keys())
short_c       = [n.replace("Qwen2.5-","") for n in model_names_c]
rmse_c  = [part_c_results[n]["RMSE_vs_GPT4"]      for n in model_names_c]
mae_c   = [part_c_results[n]["MAE_vs_GPT4"]       for n in model_names_c]
mape_c  = [part_c_results[n]["MAPE_vs_GPT4_pct"]  for n in model_names_c]
comp_c  = [part_c_results[n]["composite_score"]   for n in model_names_c]
gap_c   = [part_c_results[n]["gap_from_gpt4"]     for n in model_names_c]
gpt4_comp = part_c_results[model_names_c[0]]["gpt4_composite"]

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Qwen 2.5 — Per-Model Error Metrics vs GPT-4 Baseline\n"
             "(normalised across all 12 metrics, higher-is-better direction)",
             fontsize=13, fontweight="bold", y=1.02)

# RMSE per model
ax = axes[0]
bars = ax.bar(short_c, rmse_c, color=PALETTE, edgecolor=AMBER_D, linewidth=0.8, alpha=0.9)
for bar, v in zip(bars, rmse_c):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f"{v:.2f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("RMSE (normalised score units)", fontsize=10)
ax.set_title("RMSE vs GPT-4\n(lower = closer to GPT-4 quality)", fontsize=10, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)

# MAE per model
ax = axes[1]
bars = ax.bar(short_c, mae_c, color=PALETTE, edgecolor=AMBER_D, linewidth=0.8, alpha=0.9)
for bar, v in zip(bars, mae_c):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{v:.2f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("MAE (normalised score units)", fontsize=10)
ax.set_title("MAE vs GPT-4\n(robust to extreme metrics like response_time)",
             fontsize=10, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Composite + GPT-4 line
ax = axes[2]
bars = ax.bar(short_c, comp_c, color=PALETTE, edgecolor=AMBER_D, linewidth=0.8, alpha=0.9)
ax.axhline(gpt4_comp, color="#10b981", linestyle="--", linewidth=2, label=f"GPT-4 baseline ({gpt4_comp:.1f})")
for bar, v, g in zip(bars, comp_c, gap_c):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{v:.1f}\n({g:+.1f})", ha="center", fontsize=8.5, fontweight="bold")
ax.set_ylabel("Composite Score (%)", fontsize=10)
ax.set_title("Composite Score vs GPT-4 Baseline\n(gap shown in parentheses)",
             fontsize=10, fontweight="bold")
ax.set_ylim(0, 110); ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
out = os.path.join(OUT_DIR, "regression_per_model_vs_gpt4.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"Plot saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLE TABLES
# ═══════════════════════════════════════════════════════════════════════════════
SEP = "=" * 110
sep = "-" * 110

print(f"\n{SEP}")
print("PART A — All 19 LLMs: Scaling Law Regression Summary")
print(SEP)
for bench, key in [("MMLU","MMLU_scaling_law"),("HumanEval","HumanEval_scaling_law")]:
    d = part_a_results[key]
    print(f"\n  {bench}:  {d['formula']}")
    print(f"  R²={d['R2']:.4f}  Adj-R²={d['Adjusted_R2']:.4f}  RMSE={d['RMSE']:.4f}  "
          f"MSE={d['MSE']:.4f}  MAE={d['MAE']:.4f}  MAPE={d['MAPE_pct']:.2f}%  Pearson r={d['Pearson_r']:.4f}")
print(f"\n  {'Model':<22} {'MMLU Act':>9} {'MMLU Pred':>10} {'Residual':>10}   {'HEval Act':>10} {'HEval Pred':>11} {'Residual':>10}")
print(f"  {'-'*85}")
for m in ALL_MODELS:
    dm = part_a_results["MMLU_scaling_law"]["per_model"][m["name"]]
    dh = part_a_results["HumanEval_scaling_law"]["per_model"][m["name"]]
    mark = " *" if m["family"]=="Qwen2.5" else "  "
    print(f"  {m['name']+mark:<22} {dm['actual']:>9.1f} {dm['predicted']:>10.2f} {dm['residual']:>+10.2f}   "
          f"{dh['actual']:>10.1f} {dh['predicted']:>11.2f} {dh['residual']:>+10.2f}")
print(SEP)

print(f"\n{SEP}")
print("PART B — Qwen2.5: Regression Metrics per Metric (log-linear fit vs params_B)")
print(SEP)
print(f"  {'Metric':<30} {'R²':>7} {'Adj-R²':>8} {'RMSE':>7} {'MSE':>8} {'MAE':>7} {'MAPE%':>7} {'Pearson r':>10}")
print(f"  {'-'*90}")
for key in metric_keys:
    d = part_b_results[key]
    lib = " ▼" if d["lower_is_better"] else "  "
    print(f"  {key.replace('_',' ').title()+lib:<30} {d['R2']:>7.4f} {d['Adjusted_R2']:>8.4f} "
          f"{d['RMSE']:>7.4f} {d['MSE']:>8.4f} {d['MAE']:>7.4f} {d['MAPE_pct']:>7.2f} {d['Pearson_r']:>10.4f}")
print(SEP)

print(f"\n{SEP}")
print("PART C — Per-Model Error vs GPT-4 Reference (normalised, higher-is-better)")
print(SEP)
print(f"  {'Model':<18} {'RMSE':>7} {'MSE':>8} {'MAE':>7} {'MAPE%':>7} {'Composite':>10} {'GPT-4':>7} {'Gap':>8}")
print(f"  {'-'*80}")
for name in model_names_c:
    d = part_c_results[name]
    print(f"  {name:<18} {d['RMSE_vs_GPT4']:>7.2f} {d['MSE_vs_GPT4']:>8.2f} "
          f"{d['MAE_vs_GPT4']:>7.2f} {d['MAPE_vs_GPT4_pct']:>7.2f} "
          f"{d['composite_score']:>10.2f} {d['gpt4_composite']:>7.2f} {d['gap_from_gpt4']:>+8.2f}")
print(SEP)

print(f"\n--- OUTPUT FILES ---")
for f in ["qwen25_regression_metrics.json","regression_all19_scaling.png",
          "regression_qwen25_fit_quality.png","regression_qwen25_fit_lines.png",
          "regression_per_model_vs_gpt4.png"]:
    print(f"  {OUT_DIR}/{f}")
print(f"\nRegression metrics complete.")
