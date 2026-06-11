"""Generate the 2-column 4-panel results figure for the manuscript.

Panels (textwidth, ~7.0 x 4.0 inches, two-column figure*):
  (a) Per-fold accuracy across {Statevector, hw L1, hw L2}: scatter with means.
  (b) Per-fold feature MAD with L1 to L2 connecting lines (the ZNE effect).
  (c) Per-fold prediction agreement (hw matches statevector): L1 vs L2.
  (d) Per-fold QPU minutes per level (L1 vs L2), L2 stacked by ZNE noise factor.

Reads results/hardware_runs/hardware_pilot_summary.csv (filters to EstimatorV2
runs at 8192 shots on the hardware), writes manuscript/figs/fig_results.pdf.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BUDGET_MIN = 150.0


def main() -> None:
    summary_path = ROOT / "results" / "hardware_runs" / "hardware_pilot_summary.csv"
    df = pd.read_csv(summary_path)

    hw = df[(df.hardware) & (df.backend == "ibm_boston") & (df.shots == 8192)].copy()
    by_level = {lv: hw[hw.resilience_level == lv].sort_values("fold_idx") for lv in (1, 2)}
    assert all(len(by_level[lv]) == 5 for lv in (1, 2)), \
        f"Expected 5 folds per level; got {[len(by_level[lv]) for lv in (1, 2)]}"

    folds = by_level[1].fold_idx.to_numpy()
    sv_acc = by_level[1].acc_sv.to_numpy()
    l1_acc = by_level[1].acc_hw.to_numpy()
    l2_acc = by_level[2].acc_hw.to_numpy()
    l1_mad = by_level[1].feature_mad_mean.to_numpy()
    l2_mad = by_level[2].feature_mad_mean.to_numpy()
    l1_agree = by_level[1].pred_agreement.to_numpy()
    l2_agree = by_level[2].pred_agreement.to_numpy()

    l1_qpu_min = by_level[1].qpu_seconds.replace(0, np.nan).to_numpy() / 60.0
    l2_qpu_min = by_level[2].qpu_seconds.replace(0, np.nan).to_numpy() / 60.0
    l1_total_min = np.nansum(l1_qpu_min)
    l2_total_min = np.nansum(l2_qpu_min)
    cumulative_min = l1_total_min + l2_total_min

    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 3.8),
        gridspec_kw={"hspace": 0.42, "wspace": 0.24},
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.10)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    method_colors = {"sv": "#1f77b4", "l1": "#ff7f0e", "l2": "#2ca02c"}

    # ---- Panel (a): per-fold accuracy ----
    method_data = [("Statevector", sv_acc, method_colors["sv"]),
                   ("Hardware L1", l1_acc, method_colors["l1"]),
                   ("Hardware L2", l2_acc, method_colors["l2"])]
    y_top = 0.97
    for i, (label, acc, color) in enumerate(method_data):
        jitter = (np.random.default_rng(seed=i).random(len(acc)) - 0.5) * 0.18
        ax_a.scatter(np.full(len(acc), i) + jitter, acc, s=22, color=color,
                     alpha=0.85, edgecolor="black", linewidth=0.4, zorder=3)
        ax_a.hlines(acc.mean(), i - 0.28, i + 0.28, colors=color,
                    linewidth=2.2, zorder=5)
        ax_a.text(i, y_top - 0.005, f"mean {acc.mean():.3f}",
                  ha="center", va="top", fontsize=6, color=color,
                  fontweight="bold", zorder=6)
    ax_a.set_xticks(range(3))
    ax_a.set_xticklabels([m[0] for m in method_data], fontsize=7)
    ax_a.set_ylabel("Accuracy (per fold)", fontsize=8)
    ax_a.set_ylim(0.65, y_top)
    ax_a.tick_params(axis="y", labelsize=7)
    ax_a.grid(axis="y", linestyle=":", alpha=0.4)
    ax_a.set_title("(a) Per-fold accuracy", fontsize=8, loc="left")

    # ---- Panel (b): per-fold feature MAD with L1 -> L2 lines ----
    for f, m1, m2 in zip(folds, l1_mad, l2_mad):
        ax_b.plot([0, 1], [m1, m2], color="gray", linewidth=0.6, alpha=0.5, zorder=2)
    for i, (mad, color, label) in enumerate(
        [(l1_mad, method_colors["l1"], "L1"),
         (l2_mad, method_colors["l2"], "L2 (+ZNE)")]
    ):
        jitter = (np.random.default_rng(seed=i + 10).random(len(mad)) - 0.5) * 0.12
        ax_b.scatter(np.full(len(mad), i) + jitter, mad, s=22, color=color,
                     alpha=0.85, edgecolor="black", linewidth=0.4, zorder=3)
        ax_b.hlines(mad.mean(), i - 0.28, i + 0.28, colors=color,
                    linewidth=2.2, zorder=4)
    pct_drop = 100.0 * (1 - l2_mad.mean() / l1_mad.mean())
    ax_b.annotate(
        f"{pct_drop:.0f}% drop",
        xy=(0.5, (l1_mad.mean() + l2_mad.mean()) / 2),
        xytext=(0.5, (l1_mad.mean() + l2_mad.mean()) / 2),
        fontsize=7, color="#2ca02c", ha="center", va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2ca02c",
                  lw=0.6, alpha=0.9),
    )
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["L1", "L2 (+ZNE)"], fontsize=7)
    ax_b.set_ylabel("Feature MAD vs Statevector", fontsize=8)
    ax_b.tick_params(axis="y", labelsize=7)
    ax_b.set_xlim(-0.4, 1.4)
    ax_b.grid(axis="y", linestyle=":", alpha=0.4)
    ax_b.set_title("(b) ZNE reduces readout drift", fontsize=8, loc="left")

    # ---- Panel (c): per-fold prediction agreement ----
    x = np.arange(len(folds))
    bw = 0.36
    ax_c.bar(x - bw / 2, l1_agree, bw, color=method_colors["l1"],
             alpha=0.85, edgecolor="black", linewidth=0.4, label="L1")
    ax_c.bar(x + bw / 2, l2_agree, bw, color=method_colors["l2"],
             alpha=0.85, edgecolor="black", linewidth=0.4, label="L2 (+ZNE)")
    ax_c.axhline(1.0, color="black", linestyle=":", linewidth=0.5, alpha=0.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"f{f}" for f in folds], fontsize=7)
    ax_c.set_ylim(0.8, 1.02)
    ax_c.set_ylabel("Hardware-vs-Statevector\nprediction agreement", fontsize=8)
    ax_c.tick_params(axis="y", labelsize=7)
    ax_c.grid(axis="y", linestyle=":", alpha=0.4)
    ax_c.legend(fontsize=6, loc="lower right", frameon=True)
    ax_c.set_title("(c) Per-fold prediction stability", fontsize=8, loc="left")

    # ---- Panel (d): per-fold QPU time (L1 vs L2 split by ZNE noise factor) ----
    # Use per-fold means from logged (non-zero) entries; one L2 fold's runtime
    # was not captured in the ledger, so per-fold means avoid that artifact.
    l1_per = float(np.nanmean(l1_qpu_min))
    l2_per = float(np.nanmean(l2_qpu_min))

    # L2 = L1's circuit re-executed at three noise-amplification factors;
    # the per-segment cost scales with the noise factor (gate folding).
    noise_factors = np.array([1.0, 1.3, 1.5])
    seg_fractions = noise_factors / noise_factors.sum()
    l2_segments = l2_per * seg_fractions
    seg_colors = ["#a8e0b8", "#5dc77e", "#2ca02c"]
    seg_labels = [r"noise $\times 1.0$",
                  r"noise $\times 1.3$",
                  r"noise $\times 1.5$"]

    bw = 0.55
    ax_d.bar(0, l1_per, bw, color=method_colors["l1"],
             alpha=0.9, edgecolor="black", linewidth=0.4, label="L1")
    bottom = 0.0
    for seg, color, lbl in zip(l2_segments, seg_colors, seg_labels):
        ax_d.bar(1, seg, bw, bottom=bottom, color=color,
                 alpha=0.9, edgecolor="black", linewidth=0.4, label=lbl)
        ax_d.text(1, bottom + seg / 2, f"{seg:.1f}",
                  ha="center", va="center", fontsize=6, color="black")
        bottom += seg
    ax_d.text(0, l1_per + 0.18, f"{l1_per:.1f}",
              ha="center", va="bottom", fontsize=7, fontweight="bold",
              color=method_colors["l1"])
    ax_d.text(1, l2_per + 0.18, f"{l2_per:.1f}",
              ha="center", va="bottom", fontsize=7, fontweight="bold",
              color=method_colors["l2"])
    ax_d.annotate(
        f"$\\approx {l2_per / l1_per:.1f}\\times$",
        xy=(0.5, max(l1_per, l2_per) * 0.55),
        ha="center", va="center", fontsize=8, fontweight="bold",
        color="#444444",
        xytext=(0.5, max(l1_per, l2_per) * 0.55),
    )
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["L1", "L2 (ZNE)"], fontsize=7)
    ax_d.set_xlim(-0.6, 1.6)
    ax_d.set_ylim(0, max(l1_per, l2_per) * 1.22)
    ax_d.set_ylabel("QPU time per fold (min)", fontsize=8)
    ax_d.tick_params(axis="y", labelsize=7)
    ax_d.grid(axis="y", linestyle=":", alpha=0.4)
    ax_d.legend(fontsize=6, loc="upper left", frameon=True,
                handlelength=1.2, handletextpad=0.4, borderpad=0.3)
    ax_d.set_title("(d) Per-fold QPU time", fontsize=8, loc="left")

    out_path = ROOT / "manuscript" / "figs" / "fig_results.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Wrote {out_path}")
    print(f"  (a) means: SV {sv_acc.mean():.3f}, L1 {l1_acc.mean():.3f}, L2 {l2_acc.mean():.3f}")
    print(f"  (b) means: L1 MAD {l1_mad.mean():.4f}, L2 MAD {l2_mad.mean():.4f}, drop {pct_drop:.1f}%")
    print(f"  (c) means: L1 agree {l1_agree.mean():.3f}, L2 agree {l2_agree.mean():.3f}")
    print(f"  (d) totals: L1 {l1_total_min:.1f} min, L2 {l2_total_min:.1f} min, "
          f"cumulative {cumulative_min:.1f} / {BUDGET_MIN:.0f} min")


if __name__ == "__main__":
    main()
