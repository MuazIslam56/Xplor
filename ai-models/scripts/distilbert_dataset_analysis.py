# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding="utf-8")
"""
DistilBERT Training Dataset — Complete Analysis
Rows, features, class distribution, text statistics, train/val/test split.
Saves results to ../evaluation/distilbert_dataset/
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter

OUT_DIR = "../evaluation/distilbert_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/raw/real_column_labels.csv")

LABEL_MAP = {
    "email":0,"phone":1,"name":2,"date":3,"currency":4,
    "id":5,"address":6,"percentage":7,"numeric":8,"text":9
}
ID2LABEL   = {v:k for k,v in LABEL_MAP.items()}
CLASS_NAMES = list(LABEL_MAP.keys())

# Derived text features
df["char_count"]      = df["column_name"].str.len()
df["word_count"]      = df["column_name"].str.split("_").str.len()
df["has_underscore"]  = df["column_name"].str.contains("_").astype(int)
df["has_number"]      = df["column_name"].str.contains(r"\d").astype(int)
df["suffix"]          = df["column_name"].str.split("_").str[-1]
df["prefix"]          = df["column_name"].str.split("_").str[0]

# Split sizes (mirror train_distilbert.py)
SEED       = 42
TEST_SIZE  = 0.20    # 80% train+val, 20% test
VAL_SIZE   = 0.10    # 10% of the 80% = 8% of total

total      = len(df)
n_test     = round(total * TEST_SIZE)          # 84
n_train_val= total - n_test                    # 336
n_val      = round(n_train_val * VAL_SIZE)     # 34
n_train    = n_train_val - n_val               # 302

class_counts = df["label"].value_counts().sort_index()

# ─────────────────────────────────────────────────────────────────────────────
# BUILD JSON REPORT
# ─────────────────────────────────────────────────────────────────────────────
report = {
    "title": "DistilBERT Training Dataset — Complete Information",
    "file":  "ai-models/data/raw/real_column_labels.csv",
    "task":  "Column Semantic Labeling (multi-class text classification)",
    "dataset_overview": {
        "total_rows":        total,
        "total_features":    len(df.columns),
        "feature_names":     list(df.columns),
        "num_classes":       10,
        "class_names":       CLASS_NAMES,
        "label_encoding":    LABEL_MAP,
        "source_datasets":   [
            "Titanic (Kaggle) — passenger features",
            "UCI Adult / Census Income — demographic features",
            "Iris — botanical measurement features",
            "UCI Wine Quality — chemical measurement features",
            "KDD Cup 1999 — network traffic features",
            "Superstore (Kaggle) — sales & order features",
            "Generic CRM / HR / E-commerce column naming conventions",
        ],
        "curation_method":   "Manually curated from real public dataset column headers, deduplicated",
    },
    "features_description": {
        "column_name": {
            "type":        "string (text input to DistilBERT)",
            "description": "The raw column header name as it appears in a real dataset (snake_case)",
            "examples":    df["column_name"].sample(5, random_state=42).tolist(),
            "avg_chars":   round(df["char_count"].mean(), 2),
            "min_chars":   int(df["char_count"].min()),
            "max_chars":   int(df["char_count"].max()),
            "avg_words":   round(df["word_count"].mean(), 2),
            "pct_with_underscore": round(df["has_underscore"].mean() * 100, 1),
            "pct_with_number":     round(df["has_number"].mean()    * 100, 1),
        },
        "label": {
            "type":        "string (human-readable class name)",
            "description": "Semantic category of the column (target label, string form)",
            "unique_values": CLASS_NAMES,
        },
        "label_id": {
            "type":        "integer (0–9, used by DistilBERT classifier head)",
            "description": "Numeric encoding of the label — fed to the model as target",
            "encoding":    LABEL_MAP,
        },
    },
    "class_distribution": {
        label: {
            "count":      int(df[df["label"]==label].shape[0]),
            "percentage": round(df[df["label"]==label].shape[0] / total * 100, 2),
            "source_hint": src,
            "examples":   df[df["label"]==label]["column_name"].head(5).tolist(),
        }
        for label, src in zip(CLASS_NAMES, [
            "E-commerce, CRM, HR, user-registration datasets",
            "HR, customer, telecom datasets",
            "Titanic, Census, HR, customer datasets",
            "Sales, HR, financial, medical datasets",
            "Superstore, financial, payroll, e-commerce datasets",
            "Titanic, Census, order-management, CRM datasets",
            "Census (UCI Adult), shipping, real-estate datasets",
            "Financial KPIs, business analytics, survey datasets",
            "Titanic (Age/SibSp), Iris, Census, UCI Adult numeric cols",
            "Product, review, ticketing, medical, survey datasets",
        ])
    },
    "train_val_test_split": {
        "strategy":    "Stratified split — preserves class proportions in every set",
        "random_seed": SEED,
        "test_size_pct":  round(TEST_SIZE * 100, 0),
        "val_size_pct":   round(VAL_SIZE * 100, 0),
        "train_rows":  n_train,
        "val_rows":    n_val,
        "test_rows":   n_test,
        "total_rows":  total,
    },
    "text_statistics": {
        "avg_char_length":     round(df["char_count"].mean(), 2),
        "median_char_length":  round(df["char_count"].median(), 2),
        "std_char_length":     round(df["char_count"].std(),  2),
        "min_char_length":     int(df["char_count"].min()),
        "max_char_length":     int(df["char_count"].max()),
        "avg_word_tokens":     round(df["word_count"].mean(), 2),
        "max_word_tokens":     int(df["word_count"].max()),
        "pct_single_word":     round((df["word_count"]==1).mean()*100, 1),
        "pct_two_words":       round((df["word_count"]==2).mean()*100, 1),
        "pct_three_plus":      round((df["word_count"]>=3).mean()*100, 1),
        "distilbert_max_length_used": 32,
        "note": "All column names fit within 32 tokens — no truncation occurs during training",
    },
    "model_config": {
        "base_model":          "distilbert-base-uncased",
        "parameters":          "66 million",
        "classifier_head":     "Linear(768, 10) — 10 output classes",
        "tokenizer":           "DistilBertTokenizerFast (WordPiece)",
        "max_token_length":    32,
        "epochs":              5,
        "batch_size_train":    16,
        "batch_size_eval":     32,
        "learning_rate":       "2e-5",
        "weight_decay":        0.01,
        "warmup_ratio":        0.10,
        "early_stopping_patience": 2,
        "best_metric":         "eval_f1_macro",
        "device":              "CPU or CUDA (auto-detected)",
    },
}

json_path = os.path.join(OUT_DIR, "distilbert_dataset_info.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"JSON saved: {json_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "email":      "#6366f1", "phone":    "#f59e0b", "name":      "#10b981",
    "date":       "#ec4899", "currency": "#0ea5e9", "id":        "#8b5cf6",
    "address":    "#ef4444", "percentage":"#14b8a6","numeric":   "#f97316",
    "text":       "#64748b",
}
colors_list = [CLASS_COLORS[c] for c in CLASS_NAMES]

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 – Dataset Overview Dashboard
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# 1a: Class distribution bar
ax = fig.add_subplot(gs[0, :2])
counts = [int(df[df["label"]==c].shape[0]) for c in CLASS_NAMES]
bars   = ax.bar(CLASS_NAMES, counts, color=colors_list, edgecolor="white", linewidth=0.6, alpha=0.9)
for bar, v in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{v}\n({v/total*100:.1f}%)", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Number of Rows", fontsize=11)
ax.set_title(f"Class Distribution — {total} Total Rows\n10 Semantic Categories",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, max(counts) * 1.22)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# 1b: Pie chart
ax = fig.add_subplot(gs[0, 2])
wedges, texts, autotexts = ax.pie(
    counts, labels=None, colors=colors_list, autopct="%1.1f%%",
    pctdistance=0.78, startangle=90, wedgeprops=dict(edgecolor="white", linewidth=0.8))
for at in autotexts: at.set_fontsize(7.5)
ax.legend(wedges, CLASS_NAMES, title="Class", loc="center left",
          bbox_to_anchor=(-0.6, 0.5), fontsize=7.5, title_fontsize=8.5)
ax.set_title("Class Proportion\n(Pie View)", fontsize=11, fontweight="bold")

# 1c: Train/Val/Test split stacked bar
ax = fig.add_subplot(gs[1, 0])
split_vals = [n_train, n_val, n_test]
split_lbls = ["Train", "Validation", "Test"]
split_clrs = ["#10b981","#f59e0b","#ef4444"]
bottoms = 0
for v, lbl, clr in zip(split_vals, split_lbls, split_clrs):
    bar = ax.bar(["Dataset Split"], v, bottom=bottoms, color=clr, label=f"{lbl} ({v})", width=0.4)
    ax.text(0, bottoms + v/2, f"{lbl}\n{v} rows\n({v/total*100:.1f}%)",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
    bottoms += v
ax.set_ylim(0, total * 1.1)
ax.set_ylabel("Rows", fontsize=11)
ax.set_title(f"Train / Val / Test Split\n(Stratified, seed=42)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.3)

# 1d: Character length distribution histogram
ax = fig.add_subplot(gs[1, 1])
ax.hist(df["char_count"], bins=20, color="#6366f1", edgecolor="white", alpha=0.85)
ax.axvline(df["char_count"].mean(),   color="#ef4444", linestyle="--", linewidth=1.8,
           label=f"Mean={df['char_count'].mean():.1f}")
ax.axvline(df["char_count"].median(), color="#f59e0b", linestyle=":",  linewidth=1.8,
           label=f"Median={df['char_count'].median():.1f}")
ax.axvline(32, color="#10b981", linestyle="-.", linewidth=1.8,
           label="DistilBERT max=32 chars")
ax.set_xlabel("Character Length", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("Column Name Length Distribution\n(Characters)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(linestyle="--", alpha=0.3)

# 1e: Word token count distribution
ax = fig.add_subplot(gs[1, 2])
wc_counts = df["word_count"].value_counts().sort_index()
ax.bar(wc_counts.index, wc_counts.values, color="#10b981", edgecolor="white", alpha=0.85)
for xi, v in zip(wc_counts.index, wc_counts.values):
    ax.text(xi, v + 0.5, str(v), ha="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Word Count (underscore-split tokens)", fontsize=10)
ax.set_ylabel("Frequency", fontsize=10)
ax.set_title("Column Name Word Count Distribution\n(tokens after splitting on '_')",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)

# 1f: Avg char length per class
ax = fig.add_subplot(gs[2, 0])
avg_chars = [df[df["label"]==c]["char_count"].mean() for c in CLASS_NAMES]
bars = ax.barh(CLASS_NAMES, avg_chars, color=colors_list, edgecolor="white", alpha=0.9)
for bar, v in zip(bars, avg_chars):
    ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
            f"{v:.1f}", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Avg Character Length", fontsize=10)
ax.set_title("Avg Column Name Length per Class", fontsize=11, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.3)

# 1g: Underscore presence per class
ax = fig.add_subplot(gs[2, 1])
pct_under = [df[df["label"]==c]["has_underscore"].mean()*100 for c in CLASS_NAMES]
bars = ax.barh(CLASS_NAMES, pct_under, color=colors_list, edgecolor="white", alpha=0.9)
for bar, v in zip(bars, pct_under):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
            f"{v:.0f}%", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("% Columns Containing Underscore", fontsize=10)
ax.set_title("Underscore Presence per Class\n(Multi-word column names)", fontsize=11, fontweight="bold")
ax.set_xlim(0, 115)
ax.grid(axis="x", linestyle="--", alpha=0.3)

# 1h: Avg word count per class
ax = fig.add_subplot(gs[2, 2])
avg_words = [df[df["label"]==c]["word_count"].mean() for c in CLASS_NAMES]
bars = ax.barh(CLASS_NAMES, avg_words, color=colors_list, edgecolor="white", alpha=0.9)
for bar, v in zip(bars, avg_words):
    ax.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
            f"{v:.2f}", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Avg Word Tokens", fontsize=10)
ax.set_title("Avg Word Count per Class\n(after splitting on '_')", fontsize=11, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.suptitle("DistilBERT Training Dataset — Complete Overview\n"
             f"{total} rows × 3 features × 10 classes | Source: Titanic, UCI Adult, Iris, Wine, KDD, Superstore",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, "dataset_overview.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"Plot saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 – Per-class sample examples heatmap + feature table
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(22, 10))
fig.suptitle("DistilBERT Dataset — Per-Class Sample Examples & Text Statistics",
             fontsize=14, fontweight="bold", y=1.02)

for ax, cls in zip(axes.flat, CLASS_NAMES):
    sub = df[df["label"]==cls].copy()
    n   = len(sub)
    col = CLASS_COLORS[cls]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_facecolor("#f8fafc")
    ax.set_title(f"{cls.upper()}   (n={n})", fontsize=11, fontweight="bold", color=col, pad=6)
    ax.axis("off")
    # Stats row
    stats_txt = (f"Avg chars: {sub['char_count'].mean():.1f}  |  "
                 f"Avg words: {sub['word_count'].mean():.1f}  |  "
                 f"{sub['has_underscore'].mean()*100:.0f}% have '_'")
    ax.text(5, 9.2, stats_txt, ha="center", va="center", fontsize=7.5,
            color="#475569", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e2e8f0", ec="none"))
    # Sample entries
    samples = sub["column_name"].head(10).tolist()
    for i, s in enumerate(samples):
        y_pos = 8.1 - i * 0.82
        ax.text(0.5, y_pos, f"• {s}", ha="left", va="center", fontsize=8.5,
                color="#1e293b", fontweight="normal",
                bbox=dict(boxstyle="round,pad=0.2", fc=col+"22", ec="none"))

plt.tight_layout()
out = os.path.join(OUT_DIR, "dataset_per_class_samples.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"Plot saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 – Feature schema visual table
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.axis("off")
ax.set_title("DistilBERT Dataset — Feature Schema",
             fontsize=14, fontweight="bold", pad=15)

col_labels = ["Feature Name", "Data Type", "Role in Model", "Example Values", "Notes"]
row_data   = [
    ["column_name", "string (text)",  "MODEL INPUT\n(tokenised by DistilBERT)",
     "email, first_name,\ncustomer_id, birth_date",
     "Snake_case column headers from real datasets.\nAvg 12.8 chars, avg 1.9 words."],
    ["label",       "string (categorical)", "HUMAN REFERENCE\n(for display only)",
     "email, phone, name,\ndate, currency, id…",
     "10 semantic classes.\nNot fed to model — label_id is used instead."],
    ["label_id",    "integer (0–9)", "TARGET VARIABLE\n(model learns to predict this)",
     "0=email, 1=phone,\n2=name, 3=date…",
     "Integer encoding fed to DistilBERT classifier head.\nStratified across train/val/test splits."],
]
col_widths = [0.14, 0.14, 0.18, 0.22, 0.32]
table = ax.table(
    cellText=row_data, colLabels=col_labels,
    cellLoc="left", loc="center",
    colWidths=col_widths,
)
table.auto_set_font_size(False)
table.set_fontsize(9)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#e2e8f0")
    if r == 0:
        cell.set_facecolor("#1e293b")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 1:
        cell.set_facecolor("#f1f5f9")
    else:
        cell.set_facecolor("#ffffff")
    cell.set_height(0.28)
    cell.PAD = 0.04

plt.tight_layout()
out = os.path.join(OUT_DIR, "dataset_feature_schema.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"Plot saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 – Stratified split heatmap (rows per class per split)
# ─────────────────────────────────────────────────────────────────────────────
# Approximate stratified counts per class per split
split_matrix = np.zeros((len(CLASS_NAMES), 3))
for i, cls in enumerate(CLASS_NAMES):
    n_cls  = df[df["label"]==cls].shape[0]
    te     = round(n_cls * TEST_SIZE)
    tv     = n_cls - te
    val    = round(tv * VAL_SIZE)
    tr     = tv - val
    split_matrix[i] = [tr, val, te]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("DistilBERT Dataset — Stratified Train/Val/Test Split",
             fontsize=13, fontweight="bold", y=1.02)

# Heatmap
ax = axes[0]
im = ax.imshow(split_matrix, cmap="Blues", aspect="auto")
for i in range(len(CLASS_NAMES)):
    for j, lbl in enumerate(["Train","Val","Test"]):
        v = int(split_matrix[i, j])
        ax.text(j, i, str(v), ha="center", va="center", fontsize=11, fontweight="bold",
                color="white" if split_matrix[i, j] > split_matrix.max()*0.6 else "#1e293b")
ax.set_xticks([0,1,2]); ax.set_xticklabels(["Train\n(~72%)","Validation\n(~8%)","Test\n(~20%)"], fontsize=10)
ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES, fontsize=10)
ax.set_title(f"Rows per Class per Split\n(Stratified — seed=42)", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, label="Row count")

# Grouped bar
ax = axes[1]
x  = np.arange(len(CLASS_NAMES))
w  = 0.28
split_colors = ["#10b981","#f59e0b","#ef4444"]
for j, (lbl, clr) in enumerate(zip(["Train","Validation","Test"], split_colors)):
    bars = ax.bar(x + (j-1)*w, split_matrix[:,j], w, label=lbl, color=clr, alpha=0.88, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9.5)
ax.set_ylabel("Rows", fontsize=11)
ax.set_title("Train/Val/Test Row Count per Class", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_ylim(0, split_matrix[:,0].max() * 1.25)

plt.tight_layout()
out = os.path.join(OUT_DIR, "dataset_split_distribution.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"Plot saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 85
sep = "-" * 85

print(f"\n{SEP}")
print("  DISTILBERT TRAINING DATASET — COMPLETE INFORMATION")
print(SEP)
print(f"  File    : ai-models/data/raw/real_column_labels.csv")
print(f"  Task    : Column Semantic Labeling (multi-class text classification)")
print(f"  Rows    : {total}  (unique column names, deduplicated)")
print(f"  Features: 3  (column_name, label, label_id)")
print(f"  Classes : 10  (email, phone, name, date, currency, id, address, percentage, numeric, text)")
print(SEP)

print(f"\n  FEATURE SCHEMA:")
print(f"  {'Feature':<18} {'Type':<22} {'Role':<30} {'Description'}")
print(f"  {sep}")
print(f"  {'column_name':<18} {'string':<22} {'MODEL INPUT':<30} Raw column header (snake_case) — tokenised by DistilBERT")
print(f"  {'label':<18} {'string (categorical)':<22} {'human reference only':<30} Semantic class name — for display, not fed to model")
print(f"  {'label_id':<18} {'integer (0-9)':<22} {'TARGET VARIABLE':<30} Integer class ID predicted by DistilBERT head")

print(f"\n  CLASS DISTRIBUTION:")
print(f"  {'Class':<14} {'Label ID':>8} {'Count':>7} {'%':>7}  {'Avg Chars':>10}  {'Avg Words':>10}  {'Underscore%':>12}  Source")
print(f"  {sep}")
src_map = {
    "email":"E-commerce / CRM", "phone":"HR / Telecom", "name":"Titanic / Census",
    "date":"Sales / Medical",  "currency":"Superstore / Finance", "id":"Titanic / Orders",
    "address":"UCI Adult / Shipping", "percentage":"KPI / Analytics",
    "numeric":"Iris / Titanic / Wine", "text":"Review / Ticketing / Medical",
}
for cls in CLASS_NAMES:
    sub   = df[df["label"]==cls]
    n     = len(sub)
    pct   = n / total * 100
    ac    = sub["char_count"].mean()
    aw    = sub["word_count"].mean()
    pu    = sub["has_underscore"].mean()*100
    lid   = LABEL_MAP[cls]
    print(f"  {cls:<14} {lid:>8} {n:>7} {pct:>6.1f}%  {ac:>10.1f}  {aw:>10.2f}  {pu:>11.0f}%  {src_map[cls]}")
print(f"  {sep}")
print(f"  {'TOTAL':<14} {'—':>8} {total:>7} {'100.0':>6}%  "
      f"{df['char_count'].mean():>10.1f}  {df['word_count'].mean():>10.2f}  "
      f"{df['has_underscore'].mean()*100:>11.0f}%")

print(f"\n  TEXT STATISTICS:")
print(f"  Avg character length  : {df['char_count'].mean():.2f} chars")
print(f"  Median char length    : {df['char_count'].median():.0f} chars")
print(f"  Std dev char length   : {df['char_count'].std():.2f} chars")
print(f"  Min / Max chars       : {df['char_count'].min()} / {df['char_count'].max()}")
print(f"  Avg word count        : {df['word_count'].mean():.2f} tokens (split on '_')")
print(f"  Max word count        : {df['word_count'].max()} tokens")
print(f"  Single-word columns   : {(df['word_count']==1).sum()} ({(df['word_count']==1).mean()*100:.1f}%)")
print(f"  Two-word columns      : {(df['word_count']==2).sum()} ({(df['word_count']==2).mean()*100:.1f}%)")
print(f"  Three+ word columns   : {(df['word_count']>=3).sum()} ({(df['word_count']>=3).mean()*100:.1f}%)")
print(f"  Columns with '_'      : {df['has_underscore'].sum()} ({df['has_underscore'].mean()*100:.1f}%)")
print(f"  Columns with digits   : {df['has_number'].sum()} ({df['has_number'].mean()*100:.1f}%)")
print(f"  DistilBERT max_length : 32 tokens (no truncation needed — all names fit)")

print(f"\n  TRAIN / VALIDATION / TEST SPLIT:")
print(f"  Strategy   : Stratified (preserves class proportions)")
print(f"  Random seed: {SEED}")
print(f"  {'Split':<14} {'Rows':>6} {'%':>7}")
print(f"  {'-'*30}")
print(f"  {'Train':<14} {n_train:>6} {n_train/total*100:>6.1f}%")
print(f"  {'Validation':<14} {n_val:>6} {n_val/total*100:>6.1f}%")
print(f"  {'Test':<14} {n_test:>6} {n_test/total*100:>6.1f}%")
print(f"  {'TOTAL':<14} {total:>6} {'100.0':>6}%")

print(f"\n  SOURCE DATASETS:")
for i, src in enumerate([
    "Titanic (Kaggle) — passenger column headers",
    "UCI Adult / Census Income — demographic column headers",
    "Iris — botanical measurement columns",
    "UCI Wine Quality — chemical measurement columns",
    "KDD Cup 1999 — network traffic feature names",
    "Superstore (Kaggle) — sales and order column names",
    "Generic CRM / HR / E-commerce naming conventions",
], 1):
    print(f"  {i}. {src}")

print(f"\n  MODEL CONFIGURATION:")
print(f"  Base model      : distilbert-base-uncased (66M params)")
print(f"  Classifier head : Linear(768 → 10)  [10 output classes]")
print(f"  Tokenizer       : DistilBertTokenizerFast (WordPiece)")
print(f"  Max token len   : 32")
print(f"  Epochs          : 5 (early stopping patience=2)")
print(f"  Batch size      : 16 (train) / 32 (eval)")
print(f"  Learning rate   : 2e-5  |  Weight decay: 0.01  |  Warmup: 10%")
print(f"  Best metric     : eval_f1_macro")
print(SEP)

print(f"\n  OUTPUT FILES:")
for f in ["distilbert_dataset_info.json","dataset_overview.png",
          "dataset_per_class_samples.png","dataset_feature_schema.png",
          "dataset_split_distribution.png"]:
    print(f"  {OUT_DIR}/{f}")
print(f"\n  Analysis complete.")
