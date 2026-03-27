import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

print("loading data...")
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
final_features = base_cols + uncertainty_cols + engineered_cols

X = df[final_features]
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- graph 1: correlation heatmap ---
print("correlation heatmap...")
corrs = df[final_features + ['target']].corr()['target'].abs().sort_values(ascending=False)
top_feats = corrs.index[1:16].tolist()

plt.figure(figsize=(12, 10))
sns.heatmap(df[top_feats + ['target']].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap - Top 15 Features", fontsize=15)
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# --- graph 2: t-SNE projection ---
# this takes a while but looks cool
print("t-SNE (this will take ~1 min)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6, s=15)
plt.colorbar(scatter, ticks=[0, 1], label='Class (0=FP, 1=Confirmed)')
plt.title("t-SNE Projection of Kepler Dataset", fontsize=15)
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.grid(alpha=0.3)
plt.savefig("tsne_projection.png", dpi=150, bbox_inches='tight')
plt.close()

# --- graph 3: model confidence distribution ---
print("confidence distribution...")
xgb = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0)
xgb.fit(X_scaled, y)
probs = xgb.predict_proba(X_scaled)[:, 1]

plt.figure(figsize=(10, 6))
sns.histplot(probs[y == 0], color='red', label='False Positive', kde=True, bins=30, alpha=0.5)
sns.histplot(probs[y == 1], color='blue', label='Confirmed Planet', kde=True, bins=30, alpha=0.5)
plt.title("Model Confidence - Predicted Probability Distribution", fontsize=15)
plt.xlabel("P(Confirmed)")
plt.ylabel("Count")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("model_confidence.png", dpi=150, bbox_inches='tight')
plt.close()

print("\ndone:")
print("  correlation_heatmap.png")
print("  tsne_projection.png")
print("  model_confidence.png")
