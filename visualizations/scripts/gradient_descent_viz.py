import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
        self.weight_history = []   # (w1, w2) at each step
        self.bias_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, X, y, w, b):
        y_hat = self._sigmoid(X @ w + b)
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        N = len(y)
        return -(1/N) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def fit(self, X, y):
        N, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []
        self.weight_history = []
        self.bias_history = []

        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            self.losses.append(self._compute_loss(X, y, self.weights, self.bias))
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)

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


# --- data setup ---
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=300)
model.fit(X_train_sc, y_train)

weight_history = np.array(model.weight_history)  # shape (n_iter, 2)
w1_path = weight_history[:, 0]
w2_path = weight_history[:, 1]

# --- build loss surface grid around the trajectory ---
pad = 1.5
w1_range = np.linspace(w1_path.min() - pad, w1_path.max() + pad, 120)
w2_range = np.linspace(w2_path.min() - pad, w2_path.max() + pad, 120)
W1, W2 = np.meshgrid(w1_range, w2_range)

# evaluate loss on grid (fix bias at final trained value for 2D slice)
final_bias = model.bias_history[-1]
Z = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w = np.array([W1[i, j], W2[i, j]])
        Z[i, j] = model._compute_loss(X_train_sc, y_train, w, final_bias)

# --- figure: 3 panels ---
fig = plt.figure(figsize=(16, 5))
fig.suptitle("Gradient Descent on Logistic Regression (make_moons)", fontsize=13, fontweight="bold")

# --- panel 1: loss surface + trajectory (contour) ---
ax1 = fig.add_subplot(131)
levels = np.linspace(Z.min(), Z.max(), 40)
cf = ax1.contourf(W1, W2, Z, levels=levels, cmap="RdYlGn_r", alpha=0.85)
cs = ax1.contour(W1, W2, Z, levels=levels[::4], colors="white", linewidths=0.4, alpha=0.5)
plt.colorbar(cf, ax=ax1, label="Loss")

# subsample path for clarity
step = max(1, len(w1_path) // 60)
ax1.plot(w1_path[::step], w2_path[::step], "b-", lw=1.2, alpha=0.7, label="GD path")
ax1.scatter(w1_path[0], w2_path[0], color="cyan", s=80, zorder=5, label="start")
ax1.scatter(w1_path[-1], w2_path[-1], color="white", s=80, zorder=5, marker="*", label="end")

# draw arrows to show direction
for k in range(0, len(w1_path) - step, step * 5):
    ax1.annotate("", xy=(w1_path[k+step], w2_path[k+step]),
                 xytext=(w1_path[k], w2_path[k]),
                 arrowprops=dict(arrowstyle="->", color="blue", lw=1.2))

ax1.set_xlabel("w₁")
ax1.set_ylabel("w₂")
ax1.set_title("Loss Surface & GD Trajectory")
ax1.legend(fontsize=8)

# --- panel 2: 3D surface ---
ax2 = fig.add_subplot(132, projection="3d")
ax2.plot_surface(W1, W2, Z, cmap="RdYlGn_r", alpha=0.7, linewidth=0, antialiased=True)
# project path at z=min so it's visible
ax2.plot(w1_path, w2_path, model.losses, "b-", lw=1.5, label="GD path")
ax2.scatter([w1_path[0]], [w2_path[0]], [model.losses[0]], color="cyan", s=60, zorder=5)
ax2.scatter([w1_path[-1]], [w2_path[-1]], [model.losses[-1]], color="white", s=60, marker="*", zorder=5)
ax2.set_xlabel("w₁", labelpad=4)
ax2.set_ylabel("w₂", labelpad=4)
ax2.set_zlabel("Loss", labelpad=4)
ax2.set_title("Loss Surface (3D)")
ax2.view_init(elev=30, azim=-60)

# --- panel 3: loss curve ---
ax3 = fig.add_subplot(133)
ax3.plot(model.losses, color="steelblue", lw=1.8)
ax3.set_xlabel("Iteration")
ax3.set_ylabel("Binary Cross-Entropy Loss")
ax3.set_title("Loss Curve")
ax3.grid(alpha=0.3)

# annotate a few milestones
for it in [0, 50, 150, len(model.losses)-1]:
    ax3.annotate(f"{model.losses[it]:.3f}", xy=(it, model.losses[it]),
                 xytext=(it + 5, model.losses[it] + 0.01),
                 fontsize=7, color="steelblue")

plt.tight_layout()
plt.savefig("gradient_descent_viz.png", dpi=150)
plt.close()
print("Saved gradient_descent_viz.png")
