# testing if error bars alone can classify planets
# looks at individual uncertainty features and their AUC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings

warnings.filterwarnings("ignore")

DATA = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/script/kepler_clean.csv"
OUT  = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/visualizations/images/uncertainty_deep_dive.png"

df = pd.read_csv(DATA)
y = df['target'].values

# --- raw uncertainty features ---
unc_cols = [c for c in df.columns if ('err1' in c or 'err2' in c) and c != 'target']

# --- relative uncertainty features (error / |measurement|) ---
base_map = {
    'koi_period_err1':   'koi_period',
    'koi_period_err2':   'koi_period',
    'koi_duration_err1': 'koi_duration',
    'koi_duration_err2': 'koi_duration',
    'koi_depth_err1':    'koi_depth',
    'koi_depth_err2':    'koi_depth',
    'koi_prad_err1':     'koi_prad',
    'koi_prad_err2':     'koi_prad',
    'koi_impact_err1':   'koi_impact',
    'koi_impact_err2':   'koi_impact',
    'koi_insol_err1':    'koi_insol',
    'koi_insol_err2':    'koi_insol',
    'koi_steff_err1':    'koi_steff',
    'koi_steff_err2':    'koi_steff',
    'koi_slogg_err1':    'koi_slogg',
    'koi_slogg_err2':    'koi_slogg',
    'koi_srad_err1':     'koi_srad',
    'koi_srad_err2':     'koi_srad',
}

rel_data = {}
for err_col, base_col in base_map.items():
    if err_col in df.columns and base_col in df.columns:
        rel = df[err_col].abs() / df[base_col].abs().replace(0, np.nan)
        rel = rel.fillna(rel.median())
        rel_data[f"rel_{err_col}"] = rel.values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- single-feature AUC for each raw uncertainty feature ---
print("Computing single-feature AUC for each uncertainty feature...")
raw_aucs = {}
for col in unc_cols:
    x = df[col].fillna(df[col].median()).values.reshape(-1, 1)
    xgb = XGBClassifier(n_estimators=30, eval_metric='logloss', random_state=42, verbosity=0)
    scores = cross_val_score(xgb, x, y, cv=skf, scoring='roc_auc')
    raw_aucs[col] = scores.mean()
    print(f"  {col}: {scores.mean():.4f}")

# --- single-feature AUC for relative uncertainty features ---
print("Computing single-feature AUC for relative uncertainty features...")
rel_aucs = {}
for col, vals in rel_data.items():
    x = vals.reshape(-1, 1)
    xgb = XGBClassifier(n_estimators=30, eval_metric='logloss', random_state=42, verbosity=0)
    scores = cross_val_score(xgb, x, y, cv=skf, scoring='roc_auc')
    rel_aucs[col] = scores.mean()
    print(f"  {col}: {scores.mean():.4f}")

# combine and sort
all_aucs = {**raw_aucs, **rel_aucs}
sorted_items = sorted(all_aucs.items(), key=lambda x: x[1], reverse=True)
names  = [k.replace('koi_', '').replace('_err', ' err').replace('rel_koi_', 'rel/') for k, _ in sorted_items]
aucs   = [v for _, v in sorted_items]
is_rel = ['rel_' in k for k, _ in sorted_items]

# --- feature importance from XGBoost trained on ALL uncertainty features together ---
print("Training XGBoost on all uncertainty features combined...")
X_unc = df[unc_cols].fillna(df[unc_cols].median()).values
sc = StandardScaler()
X_unc_sc = sc.fit_transform(X_unc)
xgb_full = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0)
xgb_full.fit(X_unc_sc, y)
importances = xgb_full.feature_importances_
imp_order = np.argsort(importances)[::-1]

# AUC of XGBoost trained on ALL uncertainty features (benchmark)
all_unc_auc = cross_val_score(xgb_full, X_unc_sc, y, cv=skf, scoring='roc_auc').mean()
print(f"XGBoost on ALL uncertainty features: AUC = {all_unc_auc:.4f}")

# --- confirmed vs false positive error bar distributions ---
confirmed_aucs = {col: df[df['target']==1][col].fillna(df[col].median()) for col in unc_cols[:6]}
fp_aucs        = {col: df[df['target']==0][col].fillna(df[col].median()) for col in unc_cols[:6]}

# --- figure ---
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Uncertainty Deep Dive — Can Error Bars Classify Exoplanets?\n"
             "Testing each measurement's uncertainty as a standalone classifier",
             color="white", fontsize=14, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
                       top=0.93, bottom=0.06, left=0.12, right=0.97)

# --- panel 1: single-feature AUC ranking (horizontal bar) ---
ax1 = fig.add_subplot(gs[:, 0])
ax1.set_facecolor("#0f0f1a")

bar_colors = ["#e09b5c" if r else "#5c9be0" for r in is_rel]
y_pos = np.arange(len(names))
bars = ax1.barh(y_pos, aucs, color=bar_colors, alpha=0.85)

ax1.axvline(0.5, color='white', lw=1.0, linestyle='--', alpha=0.4, label='Random (AUC=0.5)')
ax1.axvline(all_unc_auc, color='#ff6b6b', lw=1.5, linestyle='-.',
            label=f'All unc. combined: {all_unc_auc:.3f}')

for bar, auc in zip(bars, aucs):
    ax1.text(auc + 0.002, bar.get_y() + bar.get_height()/2,
             f"{auc:.3f}", va='center', color='white', fontsize=7)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(names, color='white', fontsize=7.5)
ax1.set_xlabel("5-Fold CV AUC (single feature)", color='white')
ax1.set_title("Single-Feature AUC Ranking\n(blue = raw error | orange = relative error/measurement)",
              color='white', fontsize=10)
ax1.set_xlim(0.45, max(aucs) + 0.06)
ax1.tick_params(colors='white')
ax1.grid(axis='x', alpha=0.2, color='white')
ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
for spine in ax1.spines.values():
    spine.set_edgecolor('#444')

# --- panel 2: XGBoost feature importance on all unc features ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#0f0f1a")
top_n = min(12, len(unc_cols))
top_names = [unc_cols[i].replace('koi_', '') for i in imp_order[:top_n]]
top_imps  = importances[imp_order[:top_n]]
y2 = np.arange(top_n)
ax2.barh(y2, top_imps[::-1], color="#5ccc8c", alpha=0.85)
ax2.set_yticks(y2)
ax2.set_yticklabels(top_names[::-1], color='white', fontsize=8)
ax2.set_xlabel("Feature Importance (XGBoost, trained on all unc features)", color='white', fontsize=8)
ax2.set_title(f"Top {top_n} Uncertainty Features\n(XGBoost importance when combined)", color='white', fontsize=10)
ax2.tick_params(colors='white')
ax2.grid(axis='x', alpha=0.2, color='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')

# --- panel 3: distribution of top error feature for confirmed vs FP ---
top_col = sorted_items[0][0]  # highest single-feature AUC
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("#0f0f1a")

# top feature may be a relative feature (not a df column)
if top_col in df.columns:
    col_vals = df[top_col].fillna(df[top_col].median()).values
else:
    col_vals = rel_data[top_col]

vals_confirmed = col_vals[y == 1]
vals_fp        = col_vals[y == 0]

# clip to 99th percentile for visual clarity
p99 = np.percentile(col_vals, 99)
vals_confirmed = np.clip(vals_confirmed, None, p99)
vals_fp        = np.clip(vals_fp, None, p99)

ax3.hist(vals_confirmed, bins=50, color="#5c9be0", alpha=0.6, density=True, label="Confirmed Planet")
ax3.hist(vals_fp,        bins=50, color="#e05c5c", alpha=0.6, density=True, label="False Positive")
ax3.set_xlabel(top_col.replace('koi_', ''), color='white', fontsize=9)
ax3.set_ylabel("Density", color='white')
ax3.set_title(f"Distribution of Top Uncertainty Feature\n({top_col} — AUC={all_aucs[top_col]:.3f})",
              color='white', fontsize=10)
ax3.tick_params(colors='white')
ax3.grid(alpha=0.2, color='white')
ax3.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
for spine in ax3.spines.values():
    spine.set_edgecolor('#444')

# --- panel 4: plain English interpretation ---
ax4 = fig.add_subplot(gs[2, 1])
ax4.set_facecolor("#111128")
ax4.axis('off')

top3 = sorted_items[:3]
finding_lines = []
for rank, (col, auc) in enumerate(top3, 1):
    # plain descriptions for known features
    desc_map = {
        'koi_model_snr': 'Transit signal-to-noise ratio error',
        'koi_depth_err1': 'Transit depth uncertainty (upper)',
        'koi_depth_err2': 'Transit depth uncertainty (lower)',
        'koi_prad_err1': 'Planet radius uncertainty (upper)',
        'koi_prad_err2': 'Planet radius uncertainty (lower)',
        'koi_period_err1': 'Orbital period uncertainty',
        'koi_duration_err1': 'Transit duration uncertainty',
        'koi_impact_err1': 'Impact parameter uncertainty',
        'koi_insol_err1': 'Stellar insolation uncertainty',
        'koi_steff_err1': 'Stellar temperature uncertainty',
        'koi_slogg_err1': 'Stellar surface gravity uncertainty',
        'koi_srad_err1': 'Stellar radius uncertainty',
    }
    name = desc_map.get(col, col.replace('koi_', ''))
    finding_lines.append(f"  #{rank}  {name}: AUC = {auc:.3f}")

text = (
    "Main Result\n\n"
    "Top uncertainty features by standalone AUC:\n\n"
    + "\n".join(finding_lines) + "\n\n"
    f"All uncertainty features combined: AUC = {all_unc_auc:.3f}\n\n"
    "Why? Real transiting planets produce physically repeatable brightness dips.\n"
    "NASA's pipeline fits a model and reports how well it fits (SNR, depth, duration).\n"
    "Confirmed planets have tighter, more consistent uncertainties because\n"
    "the same geometry repeats every orbit. False positives are noisier."
)
ax4.text(0.04, 0.95, text, color='#cccccc', fontsize=8.5,
         transform=ax4.transAxes, va='top', linespacing=1.7,
         fontfamily='monospace')
ax4.text(0.04, 0.97, "", color='#e09b5c', fontsize=10, fontweight='bold',
         transform=ax4.transAxes, va='top')
for spine in ax4.spines.values():
    spine.set_edgecolor('#5c9be0')
    spine.set_linewidth(1.5)

plt.savefig(OUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"Saved {OUT}")
