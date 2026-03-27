import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import make_moons
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# quick LR scratch implementation for testing
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        N, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)
            error = y_hat - y
            dw = (1/N) * (X.T @ error)
            db = (1/N) * np.sum(error)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# --- task 1: test LR on make_moons ---
print("testing LR on make_moons...")
X_moon, y_moon = make_moons(n_samples=500, noise=0.2, random_state=42)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_moon, y_moon, test_size=0.2, random_state=42)

scaler_m = StandardScaler()
X_m_train_sc = scaler_m.fit_transform(X_m_train)
X_m_test_sc = scaler_m.transform(X_m_test)

lr_moon = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
lr_moon.fit(X_m_train_sc, y_m_train)
moon_acc = accuracy_score(y_m_test, lr_moon.predict(X_m_test_sc))
print(f"LR Scratch on make_moons: {moon_acc:.4f}\n")


# --- task 2: 10-fold CV on kepler data ---
print("running 10-fold CV on kepler data...")
df = pd.read_csv('kepler_clean.csv')

# feature engineering
df['depth_duration_ratio'] = df['koi_depth'] / df['koi_duration'].replace(0, np.nan)
df['prad_srad_ratio'] = df['koi_prad'] / df['koi_srad'].replace(0, np.nan)
df['snr_depth_ratio'] = df['koi_model_snr'] / df['koi_depth'].replace(0, np.nan)
engineered_cols = ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']
for col in engineered_cols:
    df[col] = df[col].fillna(df[col].median())

all_cols = [c for c in df.columns if c != 'target']
uncertainty_cols = [c for c in all_cols if 'err1' in c or 'err2' in c]
base_cols = [c for c in all_cols if c not in uncertainty_cols and c not in engineered_cols]

feature_sets = {
    "Raw Only": base_cols,
    "Raw + Unc": base_cols + uncertainty_cols,
    "Raw + Unc + Eng": base_cols + uncertainty_cols + engineered_cols
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results_summary = []

for fs_name, fs_cols in feature_sets.items():
    print(f"running: {fs_name}...")
    X = df[fs_cols].values
    y = df['target'].values

    models = {
        "LR Scratch":   LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000),
        "LR Sklearn":   LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":      XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0)
    }

    metrics = {m: {'acc': [], 'f1': [], 'auc': []} for m in models}

    for train_idx, test_idx in skf.split(X, y):
        # scale inside fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        for m_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if m_name == "LR Scratch":
                y_prob = model.predict_proba(X_test)
            else:
                y_prob = model.predict_proba(X_test)[:, 1]
            metrics[m_name]['acc'].append(accuracy_score(y_test, y_pred))
            metrics[m_name]['f1'].append(f1_score(y_test, y_pred))
            metrics[m_name]['auc'].append(roc_auc_score(y_test, y_prob))

    for m_name in models:
        results_summary.append({
            'Model': m_name,
            'Feature Set': fs_name,
            'Avg Accuracy': np.mean(metrics[m_name]['acc']),
            'Avg F1': np.mean(metrics[m_name]['f1']),
            'Avg AUC': np.mean(metrics[m_name]['auc'])
        })

results_df = pd.DataFrame(results_summary)
results_df.to_csv('experiment_results.csv', index=False)

print("\nresults:")
print(results_df.to_string(index=False))
print("\nsaved to experiment_results.csv")
