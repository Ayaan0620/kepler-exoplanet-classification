# statistical significance tests (mcnemar + wilcoxon)
# checks if differences between models are real or just noise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

DATA = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/script/kepler_clean.csv"
OUT  = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/visualizations/images/statistical_significance.png"

# --- data ---
df = pd.read_csv(DATA)
df['depth_duration_ratio'] = df['koi_depth'] / df['koi_duration'].replace(0, np.nan)
df['prad_srad_ratio']      = df['koi_prad']  / df['koi_srad'].replace(0, np.nan)
df['snr_depth_ratio']      = df['koi_model_snr'] / df['koi_depth'].replace(0, np.nan)
for col in ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']:
    df[col] = df[col].fillna(df[col].median())

all_cols  = [c for c in df.columns if c != 'target']
unc_cols  = [c for c in all_cols if 'err1' in c or 'err2' in c]
eng_cols  = ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']
base_cols = [c for c in all_cols if c not in unc_cols and c not in eng_cols]
features  = base_cols + unc_cols + eng_cols

X = df[features].values
y = df['target'].values

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def lr_scratch(X_tr, y_tr, X_te, lr=0.01, n_iter=1000):
    N, nf = X_tr.shape
    w, b = np.zeros(nf), 0.0
    for _ in range(n_iter):
        yh = sigmoid(X_tr @ w + b)
        err = yh - y_tr
        w -= lr * (1/N) * (X_tr.T @ err)
        b -= lr * (1/N) * np.sum(err)
    return sigmoid(X_te @ w + b)

# --- 10-fold CV — collect per-fold AUC and per-sample predictions ---
print("Running 10-fold CV (this takes ~2 min)...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
model_names = ["LR (Scratch)", "LR (Sklearn)", "Random Forest", "XGBoost"]

fold_aucs   = {m: [] for m in model_names}
all_preds   = {m: np.zeros(len(y)) for m in model_names}  # predicted class
all_correct = {m: np.zeros(len(y), dtype=bool) for m in model_names}

for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    # LR Scratch
    prob = lr_scratch(X_tr_sc, y_tr, X_te_sc)
    fold_aucs["LR (Scratch)"].append(roc_auc_score(y_te, prob))
    pred = (prob >= 0.5).astype(int)
    all_preds["LR (Scratch)"][te_idx]   = pred
    all_correct["LR (Scratch)"][te_idx] = (pred == y_te)

    # LR Sklearn
    m = LogisticRegression(max_iter=1000, random_state=42)
    m.fit(X_tr_sc, y_tr)
    prob = m.predict_proba(X_te_sc)[:, 1]
    fold_aucs["LR (Sklearn)"].append(roc_auc_score(y_te, prob))
    pred = (prob >= 0.5).astype(int)
    all_preds["LR (Sklearn)"][te_idx]   = pred
    all_correct["LR (Sklearn)"][te_idx] = (pred == y_te)

    # Random Forest
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(X_tr_sc, y_tr)
    prob = m.predict_proba(X_te_sc)[:, 1]
    fold_aucs["Random Forest"].append(roc_auc_score(y_te, prob))
    pred = (prob >= 0.5).astype(int)
    all_preds["Random Forest"][te_idx]   = pred
    all_correct["Random Forest"][te_idx] = (pred == y_te)

    # XGBoost
    m = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0)
    m.fit(X_tr_sc, y_tr)
    prob = m.predict_proba(X_te_sc)[:, 1]
    fold_aucs["XGBoost"].append(roc_auc_score(y_te, prob))
    pred = (prob >= 0.5).astype(int)
    all_preds["XGBoost"][te_idx]   = pred
    all_correct["XGBoost"][te_idx] = (pred == y_te)

    print(f"  fold {fold+1}/10 done")

# --- compute p-value matrices ---
n = len(model_names)
wilcoxon_p = np.ones((n, n))
mcnemar_p  = np.ones((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        mi, mj = model_names[i], model_names[j]

        # Wilcoxon on fold AUCs
        try:
            _, p = wilcoxon(fold_aucs[mi], fold_aucs[mj], alternative='two-sided')
            wilcoxon_p[i, j] = p
        except Exception:
            wilcoxon_p[i, j] = 1.0

        # McNemar on disagreements
        b = np.sum( all_correct[mi] & ~all_correct[mj])  # i right, j wrong
        c = np.sum(~all_correct[mi] &  all_correct[mj])  # i wrong, j right
        table = [[0, b], [c, 0]]
        try:
            result = mcnemar(table, exact=False, correction=True)
            mcnemar_p[i, j] = result.pvalue
        except Exception:
            mcnemar_p[i, j] = 1.0

# --- figure ---
def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Statistical Significance of Model Differences\n"
             "* p<0.05   ** p<0.01   *** p<0.001   ns = not significant",
             color="white", fontsize=13, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4,
                       top=0.92, bottom=0.08, left=0.07, right=0.97)

cmap = plt.cm.RdYlGn_r

# --- Wilcoxon heatmap ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#0f0f1a")
im1 = ax1.imshow(wilcoxon_p, cmap=cmap, vmin=0, vmax=0.1, aspect='auto')
ax1.set_xticks(range(n)); ax1.set_xticklabels(model_names, rotation=30, ha='right', color='white', fontsize=8)
ax1.set_yticks(range(n)); ax1.set_yticklabels(model_names, color='white', fontsize=8)
ax1.set_title("Wilcoxon Signed-Rank\n(p-values on fold AUCs)", color='white', fontsize=10)
for i in range(n):
    for j in range(n):
        val = wilcoxon_p[i, j]
        lbl = "—" if i == j else f"{val:.3f}\n{sig_label(val)}"
        ax1.text(j, i, lbl, ha='center', va='center', fontsize=7,
                 color='white' if val > 0.03 else 'black', fontweight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).ax.tick_params(colors='white')

# --- McNemar heatmap ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#0f0f1a")
im2 = ax2.imshow(mcnemar_p, cmap=cmap, vmin=0, vmax=0.1, aspect='auto')
ax2.set_xticks(range(n)); ax2.set_xticklabels(model_names, rotation=30, ha='right', color='white', fontsize=8)
ax2.set_yticks(range(n)); ax2.set_yticklabels(model_names, color='white', fontsize=8)
ax2.set_title("McNemar Test\n(p-values on per-sample correct/wrong)", color='white', fontsize=10)
for i in range(n):
    for j in range(n):
        val = mcnemar_p[i, j]
        lbl = "—" if i == j else f"{val:.3f}\n{sig_label(val)}"
        ax2.text(j, i, lbl, ha='center', va='center', fontsize=7,
                 color='white' if val > 0.03 else 'black', fontweight='bold')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(colors='white')

# --- fold AUC box plots ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("#0f0f1a")
colors_box = ["#5c9be0", "#b07de0", "#5ccc8c", "#e09b5c"]
data_box = [fold_aucs[m] for m in model_names]
bp = ax3.boxplot(data_box, patch_artist=True, notch=True,
                 medianprops=dict(color='white', lw=2))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for elem in ['whiskers', 'caps', 'fliers']:
    for item in bp[elem]:
        item.set_color('white')
ax3.set_xticklabels(model_names, rotation=20, ha='right', color='white', fontsize=8)
ax3.set_ylabel("AUC-ROC per Fold", color='white')
ax3.set_title("Fold-Level AUC Distribution\n(basis for Wilcoxon test)", color='white', fontsize=10)
ax3.tick_params(colors='white')
ax3.grid(alpha=0.2, color='white')
for spine in ax3.spines.values():
    spine.set_edgecolor('#444')

# --- plain English explanation panel ---
ax4 = fig.add_subplot(gs[1, :])
ax4.set_facecolor("#111128")
ax4.axis('off')

explanations = [
    ("What is the Wilcoxon Signed-Rank Test?",
     "Across 10 CV folds, each model gets 10 AUC scores. The Wilcoxon test asks:\n"
     "'Does model A consistently beat model B across folds, or does it vary randomly?'\n"
     "p < 0.05 means the difference is real with 95% confidence. p ≥ 0.05 means it could be chance."),
    ("What is the McNemar Test?",
     "Looks at the specific samples each model gets right vs wrong. It counts:\n"
     "  • Cases where A was right and B was wrong\n"
     "  • Cases where B was right and A was wrong\n"
     "If these are very unequal, one model is genuinely better on specific types of samples.\n"
     "p < 0.05 means the models classify meaningfully different samples correctly."),
    ("What does *** mean?",
     "p < 0.001 = less than 0.1% chance the difference is random. Very strong evidence.\n"
     "p < 0.01 (**) = less than 1% chance. Strong evidence.\n"
     "p < 0.05 (*) = less than 5% chance. Standard threshold for 'statistically significant'.\n"
     "'ns' = not significant. Could be real or noise — can't tell from this data alone."),
]

y_pos = 0.92
for title, body in explanations:
    ax4.text(0.02, y_pos, title, color="#e09b5c", fontsize=10, fontweight='bold',
             transform=ax4.transAxes)
    ax4.text(0.02, y_pos - 0.08, body, color="#cccccc", fontsize=8.5,
             transform=ax4.transAxes, verticalalignment='top',
             linespacing=1.6)
    y_pos -= 0.34

plt.savefig(OUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"Saved {OUT}")
