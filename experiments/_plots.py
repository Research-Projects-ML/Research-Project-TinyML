import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

PIPELINES = ["P→KD→QAT", "P→QAT→KD", "KD→P→QAT", "KD→QAT→P", "QAT→P→KD", "QAT→KD→P"]
COLORS    = ["steelblue", "steelblue", "goldenrod", "goldenrod", "crimson", "crimson"]

TS_ACC = np.array([
    [86.19, 86.19, 82.76, 83.85, 85.75, 83.95],  # seed 0
    [85.82, 85.95, 83.61, 83.27, 85.82, 84.02],  # seed 1
    [85.78, 86.05, 84.22, 82.32, 85.88, 85.88],  # seed 2
])
IMG_ACC = np.array([
    [79.80, 79.55, 75.69, 76.28, 79.18, 75.56],  # seed 0
    [80.11, 79.42, 77.01, 75.41, 79.59, 75.84],  # seed 1
    [79.70, 79.07, 76.79, 73.81, 79.22, 76.90],  # seed 2
])

# Baseline: mean across 3 seeds (tflite_metrics accuracy)
TS_BASE  = 87.12   # seeds: 87.38, 87.75, 86.22
IMG_BASE = 82.48   # seeds: 80.41, 83.76, 83.26

# Heatmap: mean acc per (technique, position), averaged over 2 matching pipelines × 3 seeds
TS_HEAT = np.array([
    [86.0, 84.7, 83.9],   # P   at pos 1, 2, 3
    [83.3, 85.3, 85.9],   # KD  at pos 1, 2, 3
    [85.2, 84.6, 84.7],   # QAT at pos 1, 2, 3
])
IMG_HEAT = np.array([
    [79.6, 77.9, 75.6],
    [75.8, 78.0, 79.3],
    [77.7, 77.3, 78.2],
])

# Sweep: student model, seed 0, RAM from cortex_m0plus profile
RATIOS        = [0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6]
TS_SWEEP_ACC  = [87.65, 86.66, 86.26, 85.44, 83.71, 82.59, 76.11]
TS_SWEEP_RAM  = [25.984, 23.996, 22.646, 21.109, 19.571, 18.034, 16.047]
IMG_SWEEP_ACC = [82.43, 82.81, 81.34, 78.85, 75.49, 72.03, 62.77]
IMG_SWEEP_RAM = [67.426, 59.926, 52.295, 48.320, 40.820, 36.714, 29.214]

M0_BUDGET = 20.0


def save(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved: {path}")


def plot_pruning_sweep(out_dir="results/plots"):
    fig, axes = plt.subplots(2, 1, figsize=(7, 9))
    fig.suptitle(
        "Pruning Ratio Sweep: Accuracy vs. RAM\n(seed 0, post-PTQ INT8)",
        fontsize=12, fontweight="bold"
    )

    sweep_data = [
        ("(a) Timeseries — TCN (UCI-HAR)", TS_SWEEP_ACC, TS_SWEEP_RAM),
        ("(b) Image — ResNet-8 (CIFAR-10)", IMG_SWEEP_ACC, IMG_SWEEP_RAM),
    ]

    for ax, (title, acc, ram) in zip(axes, sweep_data):
        ax2 = ax.twinx()

        line1, = ax.plot(RATIOS, acc, "o-",  color="steelblue", label="TFLite accuracy (INT8)")
        line2, = ax2.plot(RATIOS, ram, "o--", color="crimson",   label="RAM usage")
        line3  = ax2.axhline(M0_BUDGET, color="orange", linestyle="--",linewidth=1.5, label="M0+ RAM budget (20 KB)")
        ax.axvline(0.4, color="green", linestyle=":", linewidth=1.5)
        dot = ax.scatter([0.4], [acc[4]], s=120, color="green", zorder=5,label="Selected ratio (0.4)")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Pruning Ratio")
        ax.set_ylabel("TFLite Accuracy", color="steelblue")
        ax2.set_ylabel("RAM (KB)", color="crimson")
        ax.tick_params(axis="y", labelcolor="steelblue")
        ax2.tick_params(axis="y", labelcolor="crimson")

    fig.legend(handles=[line1, line2, line3, dot],loc="lower center", ncol=2,bbox_to_anchor=(0.5, -0.01), fontsize=9)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, out_dir, "pruning_sweep.png")
    save(fig, out_dir, "pruning_sweep.pdf")
    plt.show()


def plot_accuracy_bars(out_dir="results/plots"):
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    fig.suptitle(
        "Accuracy by Ordering Across Domains\n(mean ± std, n=3 seeds, post-PTQ INT8)",
        fontsize=12, fontweight="bold"
    )

    bar_data = [
        ("(a) Timeseries (UCI-HAR)", TS_ACC,  TS_BASE, (80, 90)),
        ("(b) Image (CIFAR-10)", IMG_ACC, IMG_BASE, (70, 84)),
    ]
    x = np.arange(len(PIPELINES))

    for ax, (title, acc, base, ylim) in zip(axes, bar_data):
        means = acc.mean(axis=0)
        stds  = acc.std(axis=0, ddof=1)

        ax.bar(x, means, yerr=stds, color=COLORS, capsize=4,error_kw={"elinewidth": 1.2}, width=0.6)
        ax.axhline(base, color="navy", linestyle="--", linewidth=1.3,label=f"Baseline {base:.2f}%")

        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + 0.1, f"{m:.1f}",ha="center", va="bottom", fontsize=8)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(PIPELINES, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("TFLite Accuracy (mean ± std)")
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8)

    legend_patches = [
        mpatches.Patch(facecolor="steelblue", label="P first (pos. 1)"),
        mpatches.Patch(facecolor="goldenrod", label="P middle (pos. 2)"),
        mpatches.Patch(facecolor="crimson", label="P last (pos. 3)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, out_dir, "accuracy_bars.pdf")
    save(fig, out_dir, "accuracy_bars.png")
    plt.show()


def plot_heatmaps(out_dir="results/plots"):
    ROWS = ["P", "KD", "QAT"]
    COLS = ["1st", "2nd", "3rd"]

    fig, axes = plt.subplots(2, 1, figsize=(5, 9))
    fig.suptitle(
        "Mean Accuracy by Technique Position in Sequence\n"
        "(3-stage orderings, mean across matching pipelines and seeds)",
        fontsize=11, fontweight="bold"
    )

    heat_data = [
        ("(a) Timeseries — TCN (UCI-HAR)",  TS_HEAT),
        ("(b) Image — ResNet-8 (CIFAR-10)", IMG_HEAT),
    ]

    for ax, (title, data) in zip(axes, heat_data):
        vmin, vmax = data.min() - 0.5, data.max() + 0.5
        im = ax.imshow(data, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(3))
        ax.set_xticklabels(COLS)
        ax.set_yticks(range(3))
        ax.set_yticklabels(ROWS, fontweight="bold")

        # Axis labels
        ax.set_xlabel("Position in Sequence", fontsize=10)
        ax.set_ylabel("Compression Technique", fontsize=10)

        ax.set_title(title, fontsize=10, fontweight="bold")

        # Colorbar with label
        cbar = plt.colorbar(im, ax=ax, fraction=0.04)
        cbar.set_label("Mean Accuracy (%)", fontsize=9)

        # Border box — spine visibility + linewidth
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor("black")

        # Grid lines to separate cells cleanly
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Cell annotations
        mid = vmin + 0.3 * (vmax - vmin)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{data[i, j]:.1f}%",
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if data[i, j] < mid else "black")

    plt.tight_layout()
    save(fig, out_dir, "heatmaps.png")
    save(fig, out_dir, "heatmaps.pdf")
    plt.show()


def generate_all_plots(out_dir="results/plots"):
    plot_pruning_sweep(out_dir)
    plot_accuracy_bars(out_dir)
    plot_heatmaps(out_dir)


if __name__ == "__main__":
    generate_all_plots()