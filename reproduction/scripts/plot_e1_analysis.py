"""Render E1 figures from audited descriptive tables (no training)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/e1-analysis-20261007-v1"
DEST = ROOT / "reproduction/reports/e1_20261007/figures"
DEST.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 180, "figure.facecolor": "white"})
d = pd.read_csv(SOURCE / "paired_contrasts.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
groups = [(table, model) for table in ["adult", "credit", "abalone"] for model in ["lr", "rf"]]
for ax, prevalence in zip(axes, [.1, .5]):
    for j, (gen, color) in enumerate([("CTGAN", "#2A6F97"), ("TVAE", "#C76D28")]):
        q = d[(d.comparison == "vs_real") & (d.intervention == "natural") & (d.generator == gen) & (d.prevalence == prevalence)]
        for i, (table, model) in enumerate(groups):
            values = q[(q.table == table) & (q.model == model)].delta_log_loss.to_numpy()
            x = i + (j-.5)*.23
            ax.vlines(x, values.min(), values.max(), color=color, lw=1.6)
            ax.scatter(np.repeat(x, 3)+np.array([-.035, 0, .035]), values, s=23, color=color,
                       label=gen if i == 0 else None)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(range(6), [f"{t.title()}\n{m.upper()}" for t, m in groups])
    ax.set_title(f"Synthetic replacement: {prevalence:.0%}")
    ax.set_ylabel("Log loss difference vs real reference (lower is better)")
    ax.grid(axis="y", alpha=.18)
axes[0].legend(frameon=False)
fig.suptitle("Natural synthetic data: paired effects vary by table and learner\nDots are three split seeds; lines show their range, not confidence intervals", fontsize=12)
fig.savefig(DEST / "natural_effects.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
q = d[(d.comparison == "vs_natural") & (d.intervention == "class_shuffle") & (d.prevalence == .5) & (d.table == "credit")]
for i, (generator, model) in enumerate([(g, m) for g in ["CTGAN", "TVAE"] for m in ["lr", "rf"]]):
    values = q[(q.generator == generator) & (q.model == model)].delta_log_loss.to_numpy()
    color = "#2A6F97" if model == "lr" else "#C76D28"
    ax.vlines(i, values.min(), values.max(), color=color, lw=2)
    ax.scatter(i + np.array([-.045, 0, .045]), values, color=color, s=42)
ax.axhline(0, color="black", lw=.8)
ax.set_xticks(range(4), ["CTGAN\nLR", "CTGAN\nRF", "TVAE\nLR", "TVAE\nRF"])
ax.set_ylabel("Log loss difference: class shuffle minus natural")
ax.set_title("Credit, 50% replacement: same marginal preservation, different effects\nThree split seeds; label counts and within-class feature marginals preserved")
ax.grid(axis="y", alpha=.18)
fig.savefig(DEST / "credit_structure_interaction.png")
plt.close(fig)

a = pd.read_csv(SOURCE / "all_metrics.csv")
q = a[(a.table == "abalone") & (a.generator == "TVAE") & (a.model == "rf") & (a.prevalence == .5)]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), sharey=True, layout="constrained")
for ax, seed in zip(axes, [2026, 2027, 2028]):
    r = q[q.seed == seed].set_index("intervention")
    for i, (metric, label) in enumerate([("log_loss", "Original log loss"), ("clipped_1e3_log_loss", "Clip p to [0.001, 0.999]")]):
        ax.plot([i-.12, i+.12], [r.loc["natural", metric], r.loc["label_permute", metric]], color="#888888", lw=1)
        ax.scatter(i-.12, r.loc["natural", metric], color="#2A6F97", s=55, label="Natural" if i == 0 else None)
        ax.scatter(i+.12, r.loc["label_permute", metric], color="#C76D28", s=55, label="Label permuted" if i == 0 else None)
    ax.set_xticks([0, 1], ["Original", "Clipped sensitivity"])
    ax.set_title(f"Seed {seed}; wrong endpoints\n{int(r.loc['natural', 'wrong_endpoint_count'])} natural / 0 permuted")
    ax.grid(axis="y", alpha=.18)
axes[0].set_ylabel("Real-test log loss (lower is better)")
axes[0].legend(frameon=False, loc="upper right", fontsize=8)
fig.suptitle("Abalone / TVAE / RF / 50%: much of the apparent gain is endpoint-sensitive\nClipping is a post-hoc sensitivity diagnostic, not a learned calibrator", fontsize=12)
fig.savefig(DEST / "endpoint_sensitivity.png")
plt.close(fig)
print(DEST)
