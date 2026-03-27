import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# logistic regression from scratch
class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        N = len(y)
        return -(1/N) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def fit(self, X, y):
        N, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []
        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)
            self.losses.append(self._compute_loss(y, y_hat))
            error = y_hat - y
            self.weights -= self.learning_rate * (1/N) * (X.T @ error)
            self.bias -= self.learning_rate * (1/N) * np.sum(error)
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# load data and build features
print("loading data...")
df = pd.read_csv('kepler_clean.csv')

# engineered features - ratios that might capture transit quality
df['depth_duration_ratio'] = df['koi_depth'] / df['koi_duration'].replace(0, np.nan)
df['prad_srad_ratio'] = df['koi_prad'] / df['koi_srad'].replace(0, np.nan)
df['snr_depth_ratio'] = df['koi_model_snr'] / df['koi_depth'].replace(0, np.nan)

engineered_cols = ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']
for col in engineered_cols:
    df[col] = df[col].fillna(df[col].median())

print(f"loaded. shape: {df.shape}")

# define feature sets for ablation
all_cols = [c for c in df.columns if c != 'target']
uncertainty_cols = [c for c in all_cols if 'err1' in c or 'err2' in c]
base_cols = [c for c in all_cols if c not in uncertainty_cols and c not in engineered_cols]

feature_sets = {
    "Raw features only": base_cols,
    "Raw + Uncertainty": base_cols + uncertainty_cols,
    "Raw + Unc + Engineered": base_cols + uncertainty_cols + engineered_cols
}

print("\nFeature sets:")
for name, cols in feature_sets.items():
    print(f"  {name}: {len(cols)} features")

# 10-fold CV
print("\nrunning CV... (this takes a few mins)")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []
roc_data = {}

for fs_name, fs_cols in feature_sets.items():
    print(f"  {fs_name}...")
    X = df[fs_cols].values
    y = df['target'].values

    models = {
        "LR (Scratch)": LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000),
        "LR (Sklearn)": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0)
    }

    fold_metrics = {name: {'acc': [], 'f1': [], 'auc': []} for name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # scale inside the fold - important to avoid data leakage
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        for m_name, model in models.items():
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)

            if m_name == "LR (Scratch)":
                y_prob = model.predict_proba(X_test_sc)  # 1d array
            else:
                y_prob = model.predict_proba(X_test_sc)[:, 1]

            fold_metrics[m_name]['acc'].append(accuracy_score(y_test, y_pred))
            fold_metrics[m_name]['f1'].append(f1_score(y_test, y_pred))
            fold_metrics[m_name]['auc'].append(roc_auc_score(y_test, y_prob))

            # save roc data from last fold of best feature set for plotting
            if fs_name == "Raw + Unc + Engineered" and fold_idx == 9:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_data[m_name] = (fpr, tpr, roc_auc_score(y_test, y_prob))

    for m_name in models:
        results.append({
            'Model': m_name,
            'Feature Set': fs_name,
            'Accuracy': (np.mean(fold_metrics[m_name]['acc']), np.std(fold_metrics[m_name]['acc'])),
            'F1': (np.mean(fold_metrics[m_name]['f1']), np.std(fold_metrics[m_name]['f1'])),
            'AUC-ROC': (np.mean(fold_metrics[m_name]['auc']), np.std(fold_metrics[m_name]['auc']))
        })

# results
print("\nresults:")
header = f"{'Model':<15} | {'Feature Set':<22} | {'Accuracy':<18} | {'F1':<15} | {'AUC-ROC'}"
print(header)
print("-" * 90)

prev_fs = ""
for res in results:
    if prev_fs != "" and res['Feature Set'] != prev_fs:
        print("-" * 90)
    acc_m, acc_s = res['Accuracy']
    f1_m, f1_s = res['F1']
    auc_m, auc_s = res['AUC-ROC']
    print(f"{res['Model']:<15} | {res['Feature Set']:<22} | "
          f"{acc_m*100:5.2f}% ± {acc_s*100:4.2f}% | "
          f"{f1_m:.3f} ± {f1_s:.3f} | "
          f"{auc_m:.3f} ± {auc_s:.3f}")
    prev_fs = res['Feature Set']

# plots
print("\ngenerating plots...")

# plot 1 - ROC curves
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, auc_val) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Raw + Unc + Engineered")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()

# plot 2 - LR training loss on full dataset
print("retraining LR scratch on full data for loss curve...")
full_cols = feature_sets["Raw + Unc + Engineered"]
X_full = df[full_cols].values
y_full = df['target'].values
X_full_sc = StandardScaler().fit_transform(X_full)

lr_final = LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000)
lr_final.fit(X_full_sc, y_full)

plt.figure(figsize=(8, 4))
plt.plot(lr_final.losses)
plt.xlabel("Iteration")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("LR Scratch - Training Loss (Full Dataset)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lr_training_loss.png", dpi=150)
plt.close()

# plot 3 - EDA
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
counts = df['target'].value_counts()
sns.barplot(x=counts.index, y=counts.values, palette="viridis")
plt.xticks([0, 1], ['False Positive', 'Confirmed Planet'])
plt.ylabel("Count")
plt.title("Class Distribution")
for i, v in enumerate(counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')

# scatter - plot confirmed first then FP on top so they're visible
plt.subplot(1, 2, 2)
confirmed = df[df['target'] == 1]
false_pos = df[df['target'] == 0]
plt.scatter(np.log1p(confirmed['koi_depth']), np.log1p(confirmed['koi_prad']),
            color='steelblue', alpha=0.3, s=10, label='Confirmed')
plt.scatter(np.log1p(false_pos['koi_depth']), np.log1p(false_pos['koi_prad']),
            color='orange', alpha=0.7, s=10, label='False Positive')
plt.xlabel("log(1 + koi_depth)")
plt.ylabel("log(1 + koi_prad)")
plt.title("Depth vs Planet Radius (Log Scale)")
plt.legend()

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.close()

# plot 4 - ablation bar chart
plot_data = []
for res in results:
    plot_data.append({
        'Model': res['Model'],
        'Feature Set': res['Feature Set'],
        'AUC-ROC': res['AUC-ROC'][0],
        'std': res['AUC-ROC'][1]
    })
plot_df = pd.DataFrame(plot_data)

models_list = plot_df['Model'].unique()
fs_list = plot_df['Feature Set'].unique()
x = np.arange(len(fs_list))
width = 0.2

plt.figure(figsize=(10, 6))
for i, model in enumerate(models_list):
    sub = plot_df[plot_df['Model'] == model]
    plt.bar(x + i*width - (len(models_list)-1)*width/2,
            sub['AUC-ROC'], width, label=model,
            yerr=sub['std'], capsize=5)

plt.xticks(x, fs_list)
plt.ylabel("Mean AUC-ROC")
plt.title("Ablation: Model Performance Across Feature Sets")
plt.ylim(0.8, 1.0)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("ablation_barplot.png", dpi=150)
plt.close()

# SHAP analysis
print("\nrunning SHAP...")
try:
    import shap
    xgb_final = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0)
    xgb_final.fit(X_full_sc, y_full)
    explainer = shap.TreeExplainer(xgb_final)
    shap_vals = explainer.shap_values(X_full_sc)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_full_sc, feature_names=full_cols, max_display=15, show=False)
    plt.title("SHAP Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=150)
    plt.close()
    print("SHAP done - saved shap_beeswarm.png")
except ImportError:
    print("shap not installed - run: pip install shap")
except Exception as e:
    print(f"SHAP failed: {e}")

print("\ndone")
