# heatmap of how LR weights change during training
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler

DATA = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/script/kepler_clean.csv"
OUT  = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/visualizations/images/weight_trajectory_heatmap.png"

# --- data ---
df = pd.read_csv(DATA)
df['depth_duration_ratio'] = df['koi_depth'] / df['koi_duration'].replace(0, np.nan)
df['prad_srad_ratio']      = df['koi_prad']  / df['koi_srad'].replace(0, np.nan)
df['snr_depth_ratio']      = df['koi_model_snr'] / df['koi_depth'].replace(0, np.nan)
for col in ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']:
    df[col] = df[col].fillna(df[col].median())

all_cols = [c for c in df.columns if c != 'target']
feature_names = all_cols  # keep all features

X = df[feature_names].values
y = df['target'].values

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
N, n_feat = X_sc.shape

# --- LR from scratch, record weights every step ---
lr = 0.01
n_iter = 800
weights = np.zeros(n_feat)
bias = 0.0
losses = []
weight_history = np.zeros((n_iter, n_feat))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

for i in range(n_iter):
    y_hat = sigmoid(X_sc @ weights + bias)
    y_hat_c = np.clip(y_hat, 1e-15, 1 - 1e-15)
    losses.append(-(1/N) * np.sum(y * np.log(y_hat_c) + (1-y) * np.log(1 - y_hat_c)))

    error = y_hat - y
    weights -= lr * (1/N) * (X_sc.T @ error)
    bias    -= lr * (1/N) * np.sum(error)
    weight_history[i] = weights.copy()

# sort features by |final weight| descending so most important are at top
order = np.argsort(np.abs(weight_history[-1]))[::-1]
sorted_names   = [feature_names[i] for i in order]
sorted_history = weight_history[:, order].T   # shape: (n_feat, n_iter)

# --- figure ---
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Logistic Regression Weight Trajectory During Gradient Descent\n(Kepler Exoplanet Dataset — features sorted by final |weight|)",
             color="white", fontsize=14, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[4, 1],
                       hspace=0.35, top=0.93, bottom=0.08, left=0.22, right=0.97)

# --- heatmap ---
ax_heat = fig.add_subplot(gs[0])
ax_heat.set_facecolor("#0f0f1a")

vmax = np.percentile(np.abs(sorted_history), 98)
im = ax_heat.imshow(sorted_history, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax,
                    extent=[0, n_iter, n_feat - 0.5, -0.5])

ax_heat.set_yticks(range(n_feat))
ax_heat.set_yticklabels(sorted_names, fontsize=7, color="white")
ax_heat.set_xlabel("Gradient Descent Iteration", color="white", fontsize=11)
ax_heat.set_title("Weight Value (Red = positive, Blue = negative)", color="white", fontsize=10)
ax_heat.tick_params(axis="x", colors="white")
for spine in ax_heat.spines.values():
    spine.set_edgecolor("#444")

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01)
cbar.ax.yaxis.set_tick_params(color="white")
cbar.ax.tick_params(colors="white")
cbar.set_label("Weight", color="white")

# draw vertical lines at key milestones
for it in [50, 100, 200, 400]:
    ax_heat.axvline(it, color="white", lw=0.7, alpha=0.4, linestyle="--")

# --- loss curve ---
ax_loss = fig.add_subplot(gs[1])
ax_loss.set_facecolor("#0f0f1a")
ax_loss.plot(losses, color="#5c9be0", lw=1.5)
ax_loss.set_xlabel("Iteration", color="white")
ax_loss.set_ylabel("Loss", color="white", fontsize=9)
ax_loss.set_title("Training Loss", color="white", fontsize=10)
ax_loss.tick_params(colors="white")
ax_loss.grid(alpha=0.2, color="white")
ax_loss.set_xlim(0, n_iter)
for it in [50, 100, 200, 400]:
    ax_loss.axvline(it, color="white", lw=0.7, alpha=0.4, linestyle="--")
for spine in ax_loss.spines.values():
    spine.set_edgecolor("#444")

plt.savefig(OUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
