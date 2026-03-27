import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings('ignore')

print("loading data...")
df = pd.read_csv('../data/kepler_clean_v2.csv')

# fill nans for this script (just need a quick confusion matrix)
df = df.fillna(df.median())

# feature engineering
df['depth_duration_ratio'] = df['koi_depth'] / df['koi_duration'].replace(0, np.nan)
df['prad_srad_ratio'] = df['koi_prad'] / df['koi_srad'].replace(0, np.nan)
df['snr_depth_ratio'] = df['koi_model_snr'] / df['koi_depth'].replace(0, np.nan)
eng_cols = ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']
for col in eng_cols:
    df[col] = df[col].fillna(df[col].median())

all_cols = [c for c in df.columns if c != 'target']
X = df[all_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print("training xgboost...")
model = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['False Positive', 'Confirmed'])
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title("Confusion Matrix - XGBoost", fontsize=14)
plt.grid(False)
plt.savefig("../results/figures/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

print("saved confusion_matrix.png")

tn, fp, fn, tp = cm.ravel()
print(f"\nTP (correct confirmed):    {tp}")
print(f"TN (correct rejected):     {tn}")
print(f"FP (wrong confirmed):      {fp}")
print(f"FN (missed planets):       {fn}")
