import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import shap
import warnings

warnings.filterwarnings('ignore')

print("loading data...")
df = pd.read_csv('kepler_clean.csv')

# feature engineering - same as rest of project
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

# scale and train
print("training XGBoost...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=final_features)

model = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0)
model.fit(X_scaled_df, y)

# SHAP - TreeExplainer is fast for tree models
print("running SHAP analysis...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled_df)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_scaled_df, plot_type="dot", show=False)
plt.title("SHAP Feature Importance (XGBoost)", fontsize=15)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()

print("saved shap_beeswarm.png")
