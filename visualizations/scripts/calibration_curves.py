# calibration curves - checks if predicted probabilities match reality
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

DATA = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/script/kepler_clean.csv"
OUT  = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/visualizations/images/calibration_curves.png"

# --- data + features ---
df = pd.read_csv(DATA)
df['depth_duration_ratio'] = df['koi_depth'] / df['koi_duration'].replace(0, np.nan)
df['prad_srad_ratio']      = df['koi_prad']  / df['koi_srad'].replace(0, np.nan)
df['snr_depth_ratio']      = df['koi_model_snr'] / df['koi_depth'].replace(0, np.nan)
for col in ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']:
    df[col] = df[col].fillna(df[col].median())

all_cols      = [c for c in df.columns if c != 'target']
unc_cols      = [c for c in all_cols if 'err1' in c or 'err2' in c]
eng_cols      = ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']
base_cols     = [c for c in all_cols if c not in unc_cols and c not in eng_cols]
final_features = base_cols + unc_cols + eng_cols

X = df[final_features].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# --- LR from scratch (needed for proba) ---
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def lr_scratch_fit(X, y, lr=0.01, n_iter=1000):
    N, nf = X.shape
    w = np.zeros(nf)
    b = 0.0
    for _ in range(n_iter):
        yh = sigmoid(X @ w + b)
        err = yh - y
        w -= lr * (1/N) * (X.T @ err)
        b -= lr * (1/N) * np.sum(err)
    return w, b

# --- train all 4 models ---
print("training models...")
w_lr, b_lr = lr_scratch_fit(X_train_sc, y_train)
proba_lr_scratch = sigmoid(X_test_sc @ w_lr + b_lr)

sk_lr = LogisticRegression(max_iter=1000, random_state=42)
sk_lr.fit(X_train_sc, y_train)
proba_sk_lr = sk_lr.predict_proba(X_test_sc)[:, 1]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sc, y_train)
proba_rf = rf.predict_proba(X_test_sc)[:, 1]

xgb = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0)
xgb.fit(X_train_sc, y_train)
proba_xgb = xgb.predict_proba(X_test_sc)[:, 1]

models_proba = {
    "LR (Scratch)":  proba_lr_scratch,
    "LR (Sklearn)":  proba_sk_lr,
    "Random Forest": proba_rf,
    "XGBoost":       proba_xgb,
}
model_colors = {
    "LR (Scratch)":  "#5c9be0",
    "LR (Sklearn)":  "#b07de0",
    "Random Forest": "#5ccc8c",
    "XGBoost":       "#e09b5c",
}

n_bins = 12

# --- figure ---
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Model Calibration — Reliability Diagrams\n"
             "Perfect calibration = diagonal. Curve above = underconfident. Curve below = overconfident.",
             color="white", fontsize=13, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35,
                       top=0.91, bottom=0.1, left=0.06, right=0.97)

for idx, (name, proba) in enumerate(models_proba.items()):
    color = model_colors[name]

    # --- reliability diagram ---
    ax_rel = fig.add_subplot(gs[0, idx])
    ax_rel.set_facecolor("#0f0f1a")

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins, strategy="uniform")

    # perfect calibration reference
    ax_rel.plot([0, 1], [0, 1], "w--", lw=1.2, alpha=0.5, label="Perfect")

    # shaded region showing over/under confidence
    ax_rel.fill_between(mean_pred, mean_pred, frac_pos,
                        alpha=0.15, color=color)
    ax_rel.plot(mean_pred, frac_pos, "o-", color=color, lw=2, ms=6, label=name)

    # ECE (Expected Calibration Error) — simple estimate
    bin_sizes = np.histogram(proba, bins=n_bins, range=(0, 1))[0]
    total = len(proba)
    ece = np.sum(np.abs(frac_pos - mean_pred) * (bin_sizes[bin_sizes > 0] / total))

    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.set_xlabel("Mean Predicted Probability", color="white", fontsize=9)
    ax_rel.set_ylabel("Fraction of Positives", color="white", fontsize=9)
    ax_rel.set_title(f"{name}\nECE = {ece:.4f}", color=color, fontsize=10, fontweight="bold")
    ax_rel.tick_params(colors="white")
    ax_rel.grid(alpha=0.2, color="white")
    for spine in ax_rel.spines.values():
        spine.set_edgecolor("#444")

    # --- confidence histogram ---
    ax_hist = fig.add_subplot(gs[1, idx])
    ax_hist.set_facecolor("#0f0f1a")

    ax_hist.hist(proba[y_test == 0], bins=30, color="#e05c5c", alpha=0.6,
                 label="False Positive", density=True)
    ax_hist.hist(proba[y_test == 1], bins=30, color="#5c9be0", alpha=0.6,
                 label="Confirmed", density=True)

    ax_hist.axvline(0.5, color="white", lw=1.0, linestyle="--", alpha=0.6)
    ax_hist.set_xlabel("P(Confirmed Planet)", color="white", fontsize=9)
    ax_hist.set_ylabel("Density", color="white", fontsize=9)
    ax_hist.set_title("Confidence Distribution", color="white", fontsize=9)
    ax_hist.tick_params(colors="white")
    ax_hist.grid(alpha=0.2, color="white")
    if idx == 0:
        ax_hist.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
    for spine in ax_hist.spines.values():
        spine.set_edgecolor("#444")

plt.savefig(OUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
