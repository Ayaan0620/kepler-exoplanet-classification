# compare our results to published papers on kepler classification
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

OUT = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/visualizations/images/literature_comparison.png"

# --- our results (from experiment_results.csv, best feature set = Raw + Unc + Eng) ---
our_models = {
    "LR (Scratch)\n[Ours]":   {"auc": 0.9136, "color": "#5c9be0", "marker": "o"},
    "LR (Sklearn)\n[Ours]":   {"auc": 0.9668, "color": "#b07de0", "marker": "o"},
    "Random Forest\n[Ours]":  {"auc": 0.9828, "color": "#5ccc8c", "marker": "o"},
    "XGBoost\n[Ours]":        {"auc": 0.9873, "color": "#e09b5c", "marker": "o"},
}

# --- published literature ---
# NOTE: McCauliff CV AUC = 0.9522 is period-stratified cross-validated.
# Their overall AUC on full train/test = 0.9991 (but this is optimistic, not CV).
# We use the conservative 0.9522 cross-validated number for fair comparison.
lit_models = {
    "Autovetter RF\n(McCauliff 2015)": {
        "auc": 0.9522, "auc_err": 0.008,
        "color": "#ff6b6b", "marker": "s",
        "note": "Random Forest\n~100 TCE features\nCV AUC ± std",
        "task": "binary",
    },
    "Robovetter\n(Coughlin 2016)": {
        "auc": 0.974, "auc_err": 0.01,
        "color": "#ffa07a", "marker": "D",
        "note": "Rule-based\ndecision tree\nDR24 catalog",
        "task": "binary",
    },
    "GPC Validator\n(Armstrong 2021)": {
        "auc": 0.999, "auc_err": 0.001,
        "color": "#ff4444", "marker": "^",
        "note": "Gaussian Process\nProbabilistic validation\nValidated 50 new planets",
        "task": "binary",
    },
    "XGB + Fuzzy\n(Iglesias-Alv. 2025)": {
        "auc": 0.855, "auc_err": 0.0,
        "color": "#ffaa44", "marker": "P",
        "note": "XGBoost + Fuzzy\n3-CLASS problem\n(harder task!)",
        "task": "3-class",
    },
}

# --- figure ---
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Our Models vs Published Literature\nKepler Exoplanet Classification (AUC-ROC)",
             color="white", fontsize=14, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.35,
                       top=0.93, bottom=0.07, left=0.07, right=0.97)

# --- main comparison bar chart ---
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor("#0f0f1a")

all_names  = list(our_models.keys()) + list(lit_models.keys())
all_aucs   = [v["auc"] for v in our_models.values()] + [v["auc"] for v in lit_models.values()]
all_errors = [0]*len(our_models) + [v["auc_err"] for v in lit_models.values()]
all_colors = [v["color"] for v in our_models.values()] + [v["color"] for v in lit_models.values()]
all_tasks  = ["binary"]*len(our_models) + [v["task"] for v in lit_models.values()]

x = np.arange(len(all_names))
bars = ax1.bar(x, all_aucs, color=all_colors, alpha=0.85,
               yerr=all_errors, capsize=5,
               error_kw=dict(ecolor='white', lw=1.5))

# hatch the 3-class bar to show it's a different/harder task
for bar, task in zip(bars, all_tasks):
    if task == "3-class":
        bar.set_hatch("////")
        bar.set_edgecolor("white")

# value labels
for bar, auc, task in zip(bars, all_aucs, all_tasks):
    label = f"{auc:.3f}" + ("\n(3-cls)" if task == "3-class" else "")
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             label, ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')

# dividing line between ours and literature
ax1.axvline(len(our_models) - 0.5, color='white', lw=1.5, linestyle='--', alpha=0.5)
ax1.text(len(our_models) - 0.5 - 2.1, 0.845, "← Our Models", color='white', fontsize=9, alpha=0.7)
ax1.text(len(our_models) - 0.5 + 0.1, 0.845, "Published Literature →", color='white', fontsize=9, alpha=0.7)

# reference lines
for auc_ref, label_ref, style in [(0.95, "0.95 threshold", "--"), (0.98, "0.98 threshold", ":")]:
    ax1.axhline(auc_ref, color='white', lw=0.8, linestyle=style, alpha=0.35)
    ax1.text(len(all_names) - 0.3, auc_ref + 0.001, label_ref, color='white', fontsize=7, alpha=0.5, ha='right')

ax1.set_xticks(x)
ax1.set_xticklabels(all_names, color='white', fontsize=8.5)
ax1.set_ylim(0.82, 1.01)
ax1.set_ylabel("Mean AUC-ROC", color='white')
ax1.set_title("AUC-ROC Comparison (our best feature set: Raw + Uncertainty + Engineered)", color='white', fontsize=10)
ax1.tick_params(colors='white')
ax1.grid(axis='y', alpha=0.2, color='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#444')

patch_3cls = mpatches.Patch(facecolor='#ffaa44', hatch='////', edgecolor='white', label='3-class task (harder)')
patch_bin  = mpatches.Patch(facecolor='#5c9be0', label='Binary task (same as ours)')
ax1.legend(handles=[patch_bin, patch_3cls], facecolor='#1a1a2e', labelcolor='white', fontsize=8)

# --- paper annotations ---
papers = [
    ("McCauliff et al. 2015\nApJ 806(1), 6", "#ff6b6b",
     "The foundational ML paper on Kepler TCE vetting.\nUsed ~100 tabular features from Kepler pipeline with RF.\n"
     "Cross-validated AUC = 0.9522 ± 0.0080 across orbital period bins.\n"
     "Our XGBoost (0.987) exceeds this on a comparable binary task."),
    ("Armstrong et al. 2021\nMNRAS 504(4), 5327", "#ff4444",
     "State-of-the-art probabilistic validation using Gaussian Process.\n"
     "Designed specifically for planet validation, not just classification.\n"
     "AUC ~0.999. Used to confirm 50 previously unvalidated Kepler planets.\n"
     "This is the ceiling for this problem — a specialised astronomer-built model."),
    ("Iglesias-Alvarez et al. 2025\nMathematics 13(23), 3796", "#ffaa44",
     "Most recent work on the KOI cumulative table (same data as ours).\n"
     "Tackles the harder 3-CLASS problem: Confirmed / Candidate / False Positive.\n"
     "AUC = 0.855 for 3-class is not directly comparable to our binary AUC.\n"
     "Also uses SHAP for explainability — directly comparable methodology to ours."),
]

for idx, (title, color, body) in enumerate(papers):
    ax = fig.add_subplot(gs[1, idx if idx < 2 else 1])
    ax.set_facecolor("#111128")
    ax.axis('off')
    ax.text(0.05, 0.92, title, color=color, fontsize=9, fontweight='bold',
            transform=ax.transAxes, va='top')
    ax.text(0.05, 0.72, body, color='#cccccc', fontsize=8,
            transform=ax.transAxes, va='top', linespacing=1.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.5)
    ax.set_visible(True)

# third paper in second row, spanning
ax_coughlin = fig.add_subplot(gs[1, 0])
ax_coughlin.set_facecolor("#111128")
ax_coughlin.axis('off')
title = "Coughlin et al. 2016\nApJS 224(1), 12"
body  = ("The 'Robovetter' — NASA's official automated vetter for Kepler DR24.\n"
         "Not a learned ML model: a hand-crafted rule tree built by domain experts.\n"
         "AUC ~0.974 (estimated comparatively). Produced the first fully automated\n"
         "Kepler planet candidate catalog from 48 months of data.\n"
         "Represents the 'expert baseline' — what trained astronomers encoded as rules.")
ax_coughlin.text(0.05, 0.92, title, color="#ffa07a", fontsize=9, fontweight='bold',
                 transform=ax_coughlin.transAxes, va='top')
ax_coughlin.text(0.05, 0.68, body, color='#cccccc', fontsize=8,
                 transform=ax_coughlin.transAxes, va='top', linespacing=1.6)
for spine in ax_coughlin.spines.values():
    spine.set_edgecolor("#ffa07a")
    spine.set_linewidth(1.5)

plt.savefig(OUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"Saved {OUT}")
