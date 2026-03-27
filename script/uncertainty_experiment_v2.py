import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


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


# load data (NaN kept on purpose)
print('loading data...')
df = pd.read_csv('kepler_clean_v2.csv')
y = df['target'].values
X_raw = df.drop('target', axis=1)

print(f'shape: {X_raw.shape[0]} rows x {X_raw.shape[1]} features')
print(f'target: {(y==1).sum()} confirmed, {(y==0).sum()} false positives')
print(f'NaN cells before any processing: {X_raw.isna().sum().sum()}')

# --- column groups ---
all_cols = list(X_raw.columns)
uncertainty_cols = [c for c in all_cols if 'err1' in c or 'err2' in c]
base_cols = [c for c in all_cols if c not in uncertainty_cols]

eng_cols = ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']
rel_unc_cols = ['period_rel_unc', 'depth_rel_unc', 'duration_rel_unc', 'prad_rel_unc']

feature_sets = {
    'Raw Only':             base_cols,
    'Uncertainty Only':     uncertainty_cols,
    'Raw + Uncertainty':    base_cols + uncertainty_cols,
    'Relative Uncertainty': uncertainty_cols + rel_unc_cols,
    'All Features':         base_cols + uncertainty_cols + eng_cols + rel_unc_cols,
}

print('\nfeature sets:')
for name, cols in feature_sets.items():
    print(f'  {name}: {len(cols)} features')

# --- 10-fold CV (preprocessing inside loop to avoid leakage) ---
print('\nrunning 20 experiments (5 feature sets x 4 models)...')
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model_colors = {
    'LR (Scratch)':  'steelblue',
    'LR (Sklearn)':  'mediumpurple',
    'Random Forest': 'mediumseagreen',
    'XGBoost':       'darkorange',
}

results = []

for fs_name, fs_cols in feature_sets.items():
    print(f'  {fs_name}...')
    fold_metrics = {m: {'acc': [], 'f1': [], 'auc': []} for m in model_colors}

    for train_idx, test_idx in skf.split(X_raw, y):
        # split
        X_train_df = X_raw.iloc[train_idx].copy()
        X_test_df  = X_raw.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test  = y[test_idx]

        # feature engineering on each split
        for df_part in [X_train_df, X_test_df]:
            df_part['depth_duration_ratio'] = (
                df_part['koi_depth'] / df_part['koi_duration'].replace(0, np.nan)
            )
            df_part['prad_srad_ratio'] = (
                df_part['koi_prad'] / df_part['koi_srad'].replace(0, np.nan)
            )
            df_part['snr_depth_ratio'] = (
                df_part['koi_model_snr'] / df_part['koi_depth'].replace(0, np.nan)
            )
            df_part['period_rel_unc'] = (
                df_part['koi_period_err1'] / df_part['koi_period'].abs().replace(0, np.nan)
            )
            df_part['depth_rel_unc'] = (
                df_part['koi_depth_err1'] / df_part['koi_depth'].abs().replace(0, np.nan)
            )
            df_part['duration_rel_unc'] = (
                df_part['koi_duration_err1'] / df_part['koi_duration'].abs().replace(0, np.nan)
            )
            df_part['prad_rel_unc'] = (
                df_part['koi_prad_err1'] / df_part['koi_prad'].abs().replace(0, np.nan)
            )

        # pick columns
        X_train_fs = X_train_df[fs_cols].copy()
        X_test_fs  = X_test_df[fs_cols].copy()

        # impute (fit on train only)
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train_fs)
        X_test_imp  = imputer.transform(X_test_fs)

        # scale
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_test_sc  = scaler.transform(X_test_imp)

        # train models
        models = {
            'LR (Scratch)':  LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000),
            'LR (Sklearn)':  LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost':       XGBClassifier(n_estimators=100, eval_metric='logloss',
                                           random_state=42, verbosity=0),
        }

        for m_name, model in models.items():
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = (model.predict_proba(X_test_sc)
                      if m_name == 'LR (Scratch)'
                      else model.predict_proba(X_test_sc)[:, 1])
            fold_metrics[m_name]['acc'].append(accuracy_score(y_test, y_pred))
            fold_metrics[m_name]['f1'].append(f1_score(y_test, y_pred))
            fold_metrics[m_name]['auc'].append(roc_auc_score(y_test, y_prob))

    for m_name in model_colors:
        results.append({
            'Model':       m_name,
            'Feature Set': fs_name,
            'acc_mean':    np.mean(fold_metrics[m_name]['acc']),
            'acc_std':     np.std(fold_metrics[m_name]['acc']),
            'f1_mean':     np.mean(fold_metrics[m_name]['f1']),
            'f1_std':      np.std(fold_metrics[m_name]['f1']),
            'auc_mean':    np.mean(fold_metrics[m_name]['auc']),
            'auc_std':     np.std(fold_metrics[m_name]['auc']),
        })

# --- print results ---
print('\n--- Results ---')
results_df = pd.DataFrame(results)
header = f"{'Model':<15} | {'Feature Set':<22} | {'Accuracy':<18} | {'F1':<15} | {'AUC-ROC'}"
print(header)
print('-' * len(header))

prev_fs = ''
for _, row in results_df.iterrows():
    if prev_fs and row['Feature Set'] != prev_fs:
        print('-' * len(header))
    print(f"{row['Model']:<15} | {row['Feature Set']:<22} | "
          f"{row['acc_mean']*100:5.2f}% ± {row['acc_std']*100:4.2f}% | "
          f"{row['f1_mean']:.3f} ± {row['f1_std']:.3f} | "
          f"{row['auc_mean']:.3f} ± {row['auc_std']:.3f}")
    prev_fs = row['Feature Set']

results_df.to_csv('experiment_results_v3.csv', index=False)
print('\nsaved to experiment_results_v3.csv')

# --- key finding ---
print('\n--- Key Finding (XGBoost AUC) ---')
xgb = results_df[results_df['Model'] == 'XGBoost'].set_index('Feature Set')
xgb_raw  = xgb.loc['Raw Only', 'auc_mean']
xgb_unc  = xgb.loc['Uncertainty Only', 'auc_mean']
xgb_both = xgb.loc['Raw + Uncertainty', 'auc_mean']
print(f'  raw only:         {xgb_raw:.4f}')
print(f'  uncertainty only: {xgb_unc:.4f}  <- error bars alone')
print(f'  raw + unc:        {xgb_both:.4f}')
gap = xgb_unc - xgb_raw
print(f'  gap (unc - raw):  {gap:+.4f}')

rf = results_df[results_df['Model'] == 'Random Forest'].set_index('Feature Set')
rf_raw = rf.loc['Raw Only', 'auc_mean']
rf_unc = rf.loc['Uncertainty Only', 'auc_mean']
rf_gap = rf_unc - rf_raw
print(f'\n  RF raw only:         {rf_raw:.4f}')
print(f'  RF uncertainty only: {rf_unc:.4f}')
print(f'  RF gap (unc - raw):  {rf_gap:+.4f}')

within_001_xgb = abs(gap) <= 0.01
within_001_rf  = abs(rf_gap) <= 0.01
print(f'\nuncertainty-only within 0.01 AUC of raw-only?')
print(f'  XGBoost: {"yes" if within_001_xgb else "no"}  (gap = {gap:+.4f})')
print(f'  Random Forest: {"yes" if within_001_rf else "no"}  (gap = {rf_gap:+.4f})')

# --- plots ---
print('\ngenerating plots...')
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

fs_order = list(feature_sets.keys())
model_order = list(model_colors.keys())
x = np.arange(len(fs_order))
n = len(model_order)
width = 0.18
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

for i, m_name in enumerate(model_order):
    sub = results_df[results_df['Model'] == m_name].set_index('Feature Set')
    means = [sub.loc[fs, 'auc_mean'] for fs in fs_order]
    stds  = [sub.loc[fs, 'auc_std']  for fs in fs_order]
    ax_left.bar(x + offsets[i], means, width, yerr=stds, capsize=4,
                label=m_name, color=model_colors[m_name], alpha=0.85)

ax_left.set_xticks(x)
ax_left.set_xticklabels(fs_order, rotation=20, ha='right', fontsize=9)
ax_left.set_ylim(0.75, 1.0)
ax_left.set_ylabel('Mean AUC-ROC')
ax_left.set_title('Feature Set Comparison (AUC-ROC by Model)')
ax_left.legend(fontsize=8)
ax_left.grid(axis='y', alpha=0.3)

highlight_fs = ['Raw Only', 'Uncertainty Only', 'Raw + Uncertainty']
xgb_sub = results_df[
    (results_df['Model'] == 'XGBoost') &
    (results_df['Feature Set'].isin(highlight_fs))
].set_index('Feature Set')

bars = ax_right.bar(
    highlight_fs,
    [xgb_sub.loc[fs, 'auc_mean'] for fs in highlight_fs],
    color=['steelblue', 'tomato', 'mediumseagreen'],
    alpha=0.85,
    yerr=[xgb_sub.loc[fs, 'auc_std'] for fs in highlight_fs],
    capsize=5,
)
for bar, fs in zip(bars, highlight_fs):
    val = xgb_sub.loc[fs, 'auc_mean']
    ax_right.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.005,
                  f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax_right.set_ylim(0.75, 1.0)
ax_right.set_ylabel('Mean AUC-ROC (XGBoost)')
ax_right.set_title('Can Error Bars Classify Planets?')
ax_right.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('uncertainty_investigation.png', dpi=150)
plt.close()
print('saved uncertainty_investigation.png')
