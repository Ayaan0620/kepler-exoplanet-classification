# shows how decision boundary changes during gradient descent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
        self.snapshots = {}   # {iteration: (weights, bias)}

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        N = len(y)
        return -(1/N) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def fit(self, X, y, save_at):
        N, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []

        for i in range(self.n_iterations):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)
            self.losses.append(self._compute_loss(y, y_hat))

            if i in save_at or i == self.n_iterations - 1:
                self.snapshots[i] = (self.weights.copy(), float(self.bias))

            error = y_hat - y
            self.weights -= self.learning_rate * (1/N) * (X.T @ error)
            self.bias    -= self.learning_rate * (1/N) * np.sum(error)

        return self


X, y = make_moons(n_samples=800, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

snap_iters = [0, 2, 5, 10, 20, 50, 100, 200, 499, 999]
model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_train_sc, y_train, save_at=set(snap_iters))

# build mesh in scaled space
x_min, x_max = X_train_sc[:, 0].min() - 0.5, X_train_sc[:, 0].max() + 0.5
y_min, y_max = X_train_sc[:, 1].min() - 0.5, X_train_sc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

# --- figure ---
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Logistic Regression Decision Boundary Evolution\n(Gradient Descent on make_moons)",
             fontsize=15, color="white", fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3,
                       left=0.05, right=0.95, top=0.93, bottom=0.12)

display_iters = snap_iters  # 10 panels → 2 rows of 4 + loss curve spanning bottom

cmap_bg  = plt.cm.RdBu
colors   = ["#e05c5c", "#5c9be0"]
labels   = ["False Positive", "Confirmed / Class 1"]

for idx, it in enumerate(display_iters):
    row, col = divmod(idx, 4)
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor("#0f0f1a")

    w, b = model.snapshots[it]
    probs = sigmoid(grid @ w + b).reshape(xx.shape)

    ax.contourf(xx, yy, probs, levels=50, cmap=cmap_bg, alpha=0.7, vmin=0, vmax=1)
    ax.contour(xx, yy, probs, levels=[0.5], colors="white", linewidths=1.5)

    for cls, color in zip([0, 1], colors):
        mask = y_train == cls
        ax.scatter(X_train_sc[mask, 0], X_train_sc[mask, 1],
                   c=color, s=12, alpha=0.7, edgecolors="none")

    loss_val = model.losses[it]
    ax.set_title(f"iter {it+1:4d}   loss={loss_val:.3f}",
                 color="white", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

# loss curve spanning full bottom row
ax_loss = fig.add_subplot(gs[2, :])
ax_loss.set_facecolor("#0f0f1a")
ax_loss.plot(model.losses, color="#5c9be0", lw=1.5)

# mark snapshot positions
for it in snap_iters:
    ax_loss.axvline(it, color="white", lw=0.6, alpha=0.35, linestyle="--")
    ax_loss.scatter(it, model.losses[it], color="white", s=25, zorder=5)

ax_loss.set_xlabel("Iteration", color="white")
ax_loss.set_ylabel("Binary Cross-Entropy Loss", color="white")
ax_loss.set_title("Loss Curve (white dashes = captured snapshots above)", color="white", fontsize=10)
ax_loss.tick_params(colors="white")
ax_loss.grid(alpha=0.2, color="white")
for spine in ax_loss.spines.values():
    spine.set_edgecolor("#444")

out = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/visualizations/images/decision_boundary_evolution.png"
plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out}")
