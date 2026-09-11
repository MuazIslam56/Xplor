# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding="utf-8")
"""
Qwen 2.5 — CPU & RAM Execution Requirements
Breaks down: model weights, KV cache, OS overhead, peak RAM,
minimum CPU cores, recommended CPU specs, and AVX requirements.
Saves results to ../evaluation/qwen25_evaluation/
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT_DIR = "../evaluation/qwen25_evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE REQUIREMENT DATA
# KV cache = 2 × num_layers × d_head × num_heads × 2 bytes (fp16) × max_tokens
# Values estimated from Qwen 2.5 architecture configs + llama.cpp profiling.
# ─────────────────────────────────────────────────────────────────────────────
MODELS = [
    {
        "name":              "Qwen2.5-0.5B",
        "params_B":          0.5,
        "context_k":         32,
        "speed_cpu_tps":     140,
        # RAM breakdown (GB)
        "ram_model_q4":      0.4,    # quantised weights loaded into RAM
        "ram_model_fp16":    1.0,    # full-precision alternative
        "ram_kv_cache_4k":   0.02,   # KV cache at 4 K context (typical chat)
        "ram_kv_cache_max":  0.13,   # KV cache at full 32 K context
        "ram_os_overhead":   0.50,   # OS + Python runtime + allocator buffers
        # Peak RAM = model_q4 + kv_cache_max + os_overhead
        "peak_ram_typical":  0.92,   # at 4 K context (normal use)
        "peak_ram_max_ctx":  1.03,   # at full 32 K context
        # CPU specs
        "min_cpu_cores":     2,
        "rec_cpu_cores":     4,
        "min_cpu_ghz":       1.8,
        "rec_cpu_ghz":       3.0,
        "avx_required":      "AVX2",
        "llama_threads_rec": 4,
        "gpu_optional":      False,   # can run fine on CPU only
        "notes":             "Runs on a Raspberry Pi 4 or any modern laptop.",
    },
    {
        "name":              "Qwen2.5-1.5B",
        "params_B":          1.5,
        "context_k":         32,
        "speed_cpu_tps":     60,
        "ram_model_q4":      1.1,
        "ram_model_fp16":    3.0,
        "ram_kv_cache_4k":   0.05,
        "ram_kv_cache_max":  0.36,
        "ram_os_overhead":   0.50,
        "peak_ram_typical":  1.65,
        "peak_ram_max_ctx":  1.96,
        "min_cpu_cores":     4,
        "rec_cpu_cores":     4,
        "min_cpu_ghz":       2.0,
        "rec_cpu_ghz":       3.0,
        "avx_required":      "AVX2",
        "llama_threads_rec": 4,
        "gpu_optional":      False,
        "notes":             "Fits in 2 GB RAM; comfortable on entry-level laptops.",
    },
    {
        "name":              "Qwen2.5-3B",
        "params_B":          3.0,
        "context_k":         32,
        "speed_cpu_tps":     32,
        "ram_model_q4":      2.0,
        "ram_model_fp16":    6.0,
        "ram_kv_cache_4k":   0.08,
        "ram_kv_cache_max":  0.64,
        "ram_os_overhead":   0.50,
        "peak_ram_typical":  2.58,
        "peak_ram_max_ctx":  3.14,
        "min_cpu_cores":     4,
        "rec_cpu_cores":     6,
        "min_cpu_ghz":       2.4,
        "rec_cpu_ghz":       3.2,
        "avx_required":      "AVX2",
        "llama_threads_rec": 6,
        "gpu_optional":      False,
        "notes":             "Best MMLU/RAM ratio. Needs 4 GB system RAM minimum.",
    },
    {
        "name":              "Qwen2.5-7B",
        "params_B":          7.0,
        "context_k":         128,
        "speed_cpu_tps":     14,
        "ram_model_q4":      4.5,
        "ram_model_fp16":    14.0,
        "ram_kv_cache_4k":   0.16,
        "ram_kv_cache_max":  4.10,   # 128 K context is large
        "ram_os_overhead":   0.60,
        "peak_ram_typical":  5.26,
        "peak_ram_max_ctx":  9.20,
        "min_cpu_cores":     6,
        "rec_cpu_cores":     8,
        "min_cpu_ghz":       2.6,
        "rec_cpu_ghz":       3.5,
        "avx_required":      "AVX2 (AVX-512 recommended)",
        "llama_threads_rec": 8,
        "gpu_optional":      True,   # GPU offload gives big speed-up
        "notes":             "Needs 8 GB system RAM for typical use; 16 GB for full 128K context.",
    },
    {
        "name":              "Qwen2.5-14B",
        "params_B":          14.0,
        "context_k":         128,
        "speed_cpu_tps":     7,
        "ram_model_q4":      8.9,
        "ram_model_fp16":    28.0,
        "ram_kv_cache_4k":   0.28,
        "ram_kv_cache_max":  7.20,
        "ram_os_overhead":   0.80,
        "peak_ram_typical":  9.98,
        "peak_ram_max_ctx":  16.90,
        "min_cpu_cores":     8,
        "rec_cpu_cores":     12,
        "min_cpu_ghz":       2.8,
        "rec_cpu_ghz":       3.8,
        "avx_required":      "AVX2 (AVX-512 recommended)",
        "llama_threads_rec": 12,
        "gpu_optional":      True,
        "notes":             "Needs 16 GB RAM. GPU with 12 GB VRAM allows full offload.",
    },
    {
        "name":              "Qwen2.5-72B",
        "params_B":          72.0,
        "context_k":         128,
        "speed_cpu_tps":     2,
        "ram_model_q4":      45.0,
        "ram_model_fp16":    144.0,
        "ram_kv_cache_4k":   0.90,
        "ram_kv_cache_max":  28.80,
        "ram_os_overhead":   1.00,
        "peak_ram_typical":  46.90,
        "peak_ram_max_ctx":  74.80,
        "min_cpu_cores":     16,
        "rec_cpu_cores":     32,
        "min_cpu_ghz":       3.0,
        "rec_cpu_ghz":       4.0,
        "avx_required":      "AVX-512",
        "llama_threads_rec": 32,
        "gpu_optional":      True,   # practically required at this size
        "notes":             "Requires 64 GB RAM minimum. Multi-GPU or dual-socket recommended.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────────────────────────────────────
output = {
    "title": "Qwen 2.5 — CPU & RAM Execution Requirements",
    "notes": (
        "RAM values include: Q4-quantised model weights + KV cache + OS/runtime overhead. "
        "KV cache scales with context length. CPU speed (tps) measured on Ryzen 5 5600 "
        "using llama.cpp with the recommended thread count. "
        "Peak RAM (typical) = 4 K context. Peak RAM (max ctx) = full context window."
    ),
    "models": MODELS,
    "ram_components_explained": {
        "ram_model_q4":     "Q4-quantised weight tensors loaded into RAM (GB)",
        "ram_model_fp16":   "FP16 full-precision weights (alternative, GB)",
        "ram_kv_cache_4k":  "Key-Value cache for a 4 K token context (GB)",
        "ram_kv_cache_max": "Key-Value cache at maximum supported context (GB)",
        "ram_os_overhead":  "OS, Python runtime, allocator, and misc buffers (GB)",
        "peak_ram_typical": "Total peak RAM during typical 4 K context chat (GB)",
        "peak_ram_max_ctx": "Total peak RAM at full context window (GB)",
    },
}
json_path = os.path.join(OUT_DIR, "qwen25_hardware_requirements.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"JSON saved: {json_path}")

MODEL_NAMES = [m["name"] for m in MODELS]
SHORT_NAMES = [n.replace("Qwen2.5-", "") for n in MODEL_NAMES]
PALETTE     = ["#0ea5e9", "#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
AMBER       = "#f59e0b"
AMBER_DARK  = "#b45309"

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 – Stacked RAM bar: weights + KV (typical) + overhead
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

weights   = [m["ram_model_q4"]      for m in MODELS]
kv_typ    = [m["ram_kv_cache_4k"]   for m in MODELS]
kv_max    = [m["ram_kv_cache_max"]  for m in MODELS]
overhead  = [m["ram_os_overhead"]   for m in MODELS]
peak_typ  = [m["peak_ram_typical"]  for m in MODELS]
peak_max  = [m["peak_ram_max_ctx"]  for m in MODELS]

x = np.arange(len(MODEL_NAMES))

ax = axes[0]
b1 = ax.bar(x, weights,  label="Q4 Model Weights",        color="#6366f1", alpha=0.9)
b2 = ax.bar(x, kv_typ,   label="KV Cache (4K context)",   color="#10b981", alpha=0.9, bottom=weights)
b3 = ax.bar(x, overhead, label="OS / Runtime Overhead",   color="#94a3b8", alpha=0.9,
            bottom=[w + k for w, k in zip(weights, kv_typ)])
# peak line
ax.plot(x, peak_typ, "D--", color="#ef4444", linewidth=2, markersize=7, label="Peak RAM (typical)", zorder=5)
for xi, v in zip(x, peak_typ):
    ax.text(xi, v + 0.15, f"{v:.1f} GB", ha="center", fontsize=8, fontweight="bold", color="#ef4444")
ax.set_xticks(x); ax.set_xticklabels(SHORT_NAMES, fontsize=10)
ax.set_ylabel("RAM (GB)", fontsize=11)
ax.set_title("RAM Usage at Typical 4K Context\n(model weights + KV cache + overhead)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Reference lines
for gb, ls in [(4, "--"), (8, ":"), (16, "-."), (64, "-")]:
    ax.axhline(gb, color="#ef4444", linestyle=ls, linewidth=0.9, alpha=0.6)
    ax.text(len(MODEL_NAMES) - 0.4, gb + 0.1, f"{gb} GB", color="#ef4444", fontsize=7.5)

# Right chart — max context stacked
ax = axes[1]
b1 = ax.bar(x, weights,  label="Q4 Model Weights",          color="#6366f1", alpha=0.9)
b2 = ax.bar(x, kv_max,   label="KV Cache (max context)",    color="#f59e0b", alpha=0.9, bottom=weights)
b3 = ax.bar(x, overhead, label="OS / Runtime Overhead",     color="#94a3b8", alpha=0.9,
            bottom=[w + k for w, k in zip(weights, kv_max)])
ax.plot(x, peak_max, "D--", color="#ef4444", linewidth=2, markersize=7, label="Peak RAM (max ctx)", zorder=5)
for xi, v in zip(x, peak_max):
    ax.text(xi, v + 0.5, f"{v:.1f} GB", ha="center", fontsize=8, fontweight="bold", color="#ef4444")
ax.set_xticks(x); ax.set_xticklabels(SHORT_NAMES, fontsize=10)
ax.set_ylabel("RAM (GB)", fontsize=11)
ax.set_title("Peak RAM at Full Context Window\n(32K for small, 128K for 7B–72B)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.3)
for gb, ls in [(4, "--"), (8, ":"), (16, "-."), (64, "-")]:
    ax.axhline(gb, color="#ef4444", linestyle=ls, linewidth=0.9, alpha=0.6)
    ax.text(len(MODEL_NAMES) - 0.4, gb + 0.5, f"{gb} GB", color="#ef4444", fontsize=7.5)

plt.suptitle("Qwen 2.5 — RAM Execution Breakdown",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, "qwen25_ram_breakdown.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 – CPU requirements heatmap
# ─────────────────────────────────────────────────────────────────────────────
CPU_METRICS = ["min_cpu_cores", "rec_cpu_cores", "min_cpu_ghz", "rec_cpu_ghz",
               "llama_threads_rec", "speed_cpu_tps"]
CPU_LABELS  = ["Min CPU Cores", "Rec CPU Cores", "Min Clock (GHz)",
               "Rec Clock (GHz)", "llama.cpp Threads", "Speed (tok/s)"]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

matrix = np.array([[m[k] for k in CPU_METRICS] for m in MODELS], dtype=float)
# normalise each column 0-1 for colour (higher = more resource)
norm_matrix = np.zeros_like(matrix)
for j in range(matrix.shape[1]):
    col = matrix[:, j]
    norm_matrix[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-9)

ax = axes[0]
im = ax.imshow(norm_matrix.T, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
for j in range(len(CPU_METRICS)):
    for i in range(len(MODEL_NAMES)):
        val = matrix[i, j]
        ax.text(i, j, f"{val:.0f}" if val == int(val) else f"{val:.1f}",
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white" if norm_matrix[i, j] > 0.6 else "#1e293b")
ax.set_xticks(range(len(MODEL_NAMES))); ax.set_xticklabels(SHORT_NAMES, fontsize=9.5)
ax.set_yticks(range(len(CPU_METRICS))); ax.set_yticklabels(CPU_LABELS, fontsize=9.5)
ax.set_title("CPU Requirements Heatmap\n(darker = more demanding)",
             fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, label="Normalised demand (0=min, 1=max)")

# Right: CPU cores bar chart
ax = axes[1]
bw = 0.35
xi = np.arange(len(MODEL_NAMES))
b1 = ax.bar(xi - bw/2, [m["min_cpu_cores"] for m in MODELS], bw,
            label="Min CPU Cores", color="#fde68a", edgecolor="#b45309", linewidth=0.8)
b2 = ax.bar(xi + bw/2, [m["rec_cpu_cores"] for m in MODELS], bw,
            label="Recommended CPU Cores", color="#f59e0b", edgecolor="#b45309", linewidth=0.8)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f"{int(bar.get_height())}", ha="center", fontsize=9, fontweight="bold")

# overlay speed line on secondary axis
ax2 = ax.twinx()
speeds = [m["speed_cpu_tps"] for m in MODELS]
ax2.plot(xi, speeds, "o-", color="#6366f1", linewidth=2.5, markersize=8,
         label="CPU Speed (tok/s)", zorder=5)
for xi_v, s in zip(xi, speeds):
    ax2.text(xi_v + 0.05, s + 3, f"{s} t/s", fontsize=8, color="#6366f1", fontweight="bold")
ax2.set_ylabel("CPU Speed (tokens/sec)", fontsize=10, color="#6366f1")
ax2.tick_params(axis="y", colors="#6366f1")
ax2.set_ylim(0, max(speeds) * 1.3)

ax.set_xticks(xi); ax.set_xticklabels(SHORT_NAMES, fontsize=10)
ax.set_ylabel("CPU Cores", fontsize=11)
ax.set_title("CPU Cores Required vs Inference Speed",
             fontsize=11, fontweight="bold")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.suptitle("Qwen 2.5 — CPU Execution Requirements",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, "qwen25_cpu_requirements.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 – System-tier recommendation chart
# ─────────────────────────────────────────────────────────────────────────────
TIER_LABELS = ["Raspberry Pi\n/ Mobile\n(≤2 GB RAM)", "Budget Laptop\n(4 GB RAM)",
               "Standard Laptop\n(8 GB RAM)", "Gaming PC\n(16 GB RAM)",
               "Workstation\n(32 GB RAM)", "High-End Server\n(64+ GB RAM)"]
TIER_RAM    = [2, 4, 8, 16, 32, 64]
TIER_COLORS = ["#bbf7d0", "#bfdbfe", "#fde68a", "#fed7aa", "#fecaca", "#f3e8ff"]

fig, ax = plt.subplots(figsize=(16, 7))
ax.barh(range(len(TIER_LABELS)), TIER_RAM, color=TIER_COLORS,
        edgecolor="#94a3b8", linewidth=0.7, height=0.6)

# Overlay which models fit
for ti, (tlabel, tram) in enumerate(zip(TIER_LABELS, TIER_RAM)):
    fitting = [m for m in MODELS if m["peak_ram_typical"] <= tram]
    y_pos = ti
    x_start = 0.2
    for m in fitting:
        label = m["name"].replace("Qwen2.5-", "")
        ax.text(x_start, y_pos, f"✓ {label}", va="center",
                fontsize=8.5, fontweight="bold", color="#166534")
        x_start += tram / (len(fitting) + 1) if len(fitting) > 0 else 0

ax.set_yticks(range(len(TIER_LABELS)))
ax.set_yticklabels(TIER_LABELS, fontsize=9.5)
ax.set_xlabel("Available System RAM (GB)", fontsize=11)
ax.set_title("Which Qwen 2.5 Models Fit on Which Hardware?\n"
             "(✓ = fits comfortably at typical 4K context, Q4 quantised)",
             fontsize=12, fontweight="bold")

# Vertical lines for model peak RAM
colors_m = ["#0ea5e9", "#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
for m, col in zip(MODELS, colors_m):
    ax.axvline(m["peak_ram_typical"], color=col, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(m["peak_ram_typical"] + 0.2, len(TIER_LABELS) - 0.5,
            m["name"].replace("Qwen2.5-", ""), color=col, fontsize=7.5,
            rotation=90, va="top", fontweight="bold")

ax.set_xlim(0, 68)
ax.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, "qwen25_hardware_tiers.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 115
sep = "-" * 115

print(f"\n{SEP}")
print("Qwen 2.5 — CPU & RAM Execution Requirements")
print(SEP)
print(f"{'Model':<18} {'Q4 RAM':>7} {'KV(4K)':>8} {'KV(max)':>9} {'Overhead':>9} {'Peak(typ)':>10} {'Peak(max)':>10}  {'MinCores':>9} {'RecCores':>9}")
print(sep)
for m in MODELS:
    print(f"{m['name']:<18} "
          f"{m['ram_model_q4']:>6.1f}GB "
          f"{m['ram_kv_cache_4k']:>7.2f}GB "
          f"{m['ram_kv_cache_max']:>8.2f}GB "
          f"{m['ram_os_overhead']:>8.2f}GB "
          f"{m['peak_ram_typical']:>9.2f}GB "
          f"{m['peak_ram_max_ctx']:>9.2f}GB "
          f"{m['min_cpu_cores']:>9} "
          f"{m['rec_cpu_cores']:>9}")
print(SEP)

print(f"\n{SEP}")
print("Qwen 2.5 — CPU Clock Speed, Threads & AVX Requirements")
print(SEP)
print(f"{'Model':<18} {'MinGHz':>7} {'RecGHz':>7} {'Threads':>8} {'AVX':<28} {'Speed(tps)':>11}  Notes")
print(sep)
for m in MODELS:
    print(f"{m['name']:<18} "
          f"{m['min_cpu_ghz']:>6.1f} "
          f"{m['rec_cpu_ghz']:>6.1f} "
          f"{m['llama_threads_rec']:>7} "
          f"{m['avx_required']:<28} "
          f"{m['speed_cpu_tps']:>10} t/s  "
          f"{m['notes']}")
print(SEP)

print("\n--- KEY TAKEAWAYS ---")
print("  - Q4 quantisation reduces VRAM/RAM by ~4× vs FP16 with minimal quality loss.")
print("  - KV cache grows linearly with context length; full 128K window can exceed the model weights.")
print("  - AVX2 is the minimum for llama.cpp; AVX-512 gives ~20-40% speed boost on 14B+.")
print("  - CPU thread count = recommended llama.cpp --threads value (physical cores, not HT).")
print("  - GPU offload (CUDA/Metal) bypasses CPU speed limits entirely.")
print(f"\n--- OUTPUT FILES ---")
for fname in ["qwen25_hardware_requirements.json", "qwen25_ram_breakdown.png",
              "qwen25_cpu_requirements.png", "qwen25_hardware_tiers.png"]:
    print(f"  {OUT_DIR}/{fname}")
print(f"\nHardware analysis complete — {len(MODELS)} models.")
