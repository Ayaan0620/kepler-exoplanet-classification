import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# our own logistic regression using gradient descent
# no sklearn used inside the class
class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        # clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y, y_hat):
        # binary cross entropy loss
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        N = len(y)
        return -(1/N) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def fit(self, X, y):
        N, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []

        for _ in range(self.n_iterations):
            # forward pass
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            self.losses.append(self._compute_loss(y, y_hat))

            # gradients
            error = y_hat - y
            dw = (1/N) * (X.T @ error)
            db = (1/N) * np.sum(error)

            # update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# --- test on make_moons dataset ---

if __name__ == '__main__':
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    our_model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    our_model.fit(X_train_sc, y_train)

    # compare with sklearn
    sk_model = LogisticRegression(max_iter=1000)
    sk_model.fit(X_train_sc, y_train)

    our_acc = accuracy_score(y_test, our_model.predict(X_test_sc))
    sk_acc = accuracy_score(y_test, sk_model.predict(X_test_sc))

    print(f"Our model accuracy:  {our_acc:.4f}")
    print(f"Sklearn accuracy:    {sk_acc:.4f}")
    print(f"Difference:          {abs(our_acc - sk_acc):.4f}")

    # print(our_model.losses[:5])  # debugging

    # check that loss goes down
    print(f"\nLoss start: {our_model.losses[0]:.4f}")
    print(f"Loss mid:   {our_model.losses[499]:.4f}")
    print(f"Loss end:   {our_model.losses[-1]:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(our_model.losses)
    plt.xlabel("Iteration")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig("../results/figures/lr_loss_curve.png", dpi=150)
    plt.close()
    print("\nSaved lr_loss_curve.png")
