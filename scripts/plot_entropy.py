#!/usr/bin/env python3
"""Plot UQ entropy distribution for the paper."""
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent

with open(BASE / "data/uncertainty_distribution.json") as f:
    d = json.load(f)

u = np.array(d["uncertainties"])
tau_H = d["tau_H"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Histogram
ax1.hist(u, bins=50, color="#1565C0", alpha=0.8, edgecolor="white", linewidth=0.5)
ax1.axvline(tau_H, color="red", linestyle="--", linewidth=2,
            label=r"$\tau_H$ = " + f"{tau_H:.2f}")
ax1.set_xlabel("Heuristic Uncertainty Score", fontsize=12)
ax1.set_ylabel("Number of Fields", fontsize=12)
ax1.set_title("UQ Distribution (ITOpsGraph, Llama)", fontsize=13)
ax1.legend(fontsize=11)

below = int(np.sum(u <= tau_H))
above = int(np.sum(u > tau_H))
total = len(u)
ax1.annotate(f"{below*100//total}% below\n(no reflection)",
             xy=(0.12, ax1.get_ylim()[1]*0.7), fontsize=10, color="#1565C0")
if above > 0:
    ax1.annotate(f"{above/total*100:.1f}% above\n(reflect)",
                 xy=(tau_H + 0.03, ax1.get_ylim()[1]*0.3), fontsize=10, color="red")

# CDF
sorted_u = np.sort(u)
cdf = np.arange(1, len(sorted_u)+1) / len(sorted_u)
ax2.plot(sorted_u, cdf, color="#1565C0", linewidth=2)
ax2.axvline(tau_H, color="red", linestyle="--", linewidth=2,
            label=r"$\tau_H$ = " + f"{tau_H:.2f}")
ax2.axhline(below/total, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Uncertainty Score", fontsize=12)
ax2.set_ylabel("Cumulative Fraction", fontsize=12)
ax2.set_title(f"CDF: Gate fires on {above/total*100:.1f}% of fields", fontsize=13)
ax2.legend(fontsize=11)

plt.tight_layout()
outpath = BASE / "paper" / "charts" / "chart_entropy_distribution.png"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=200, bbox_inches="tight")
print(f"Saved: {outpath}")

print(f"\nPaper stats:")
print(f"  Total fields: {total}")
print(f"  tau_H: {tau_H}")
print(f"  Gate firing rate: {above}/{total} ({above/total*100:.1f}%)")
print(f"  Mean uncertainty: {np.mean(u):.4f}")
print(f"  Confident (<=0.15): {int(np.sum(u <= 0.15))}/{total} ({np.sum(u <= 0.15)/total*100:.1f}%)")
