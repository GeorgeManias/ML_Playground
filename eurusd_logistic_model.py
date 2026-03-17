import numpy as np
import matplotlib.pyplot as plt
from data_cleaner_excel import load_data

# Graph Function
def plot_decision_boundary(X, y, weights, bias, mu, sigma):
    # scatter plot
    plt.scatter(X[:, 0], X[:, 1], c=y)

    # x values for the line
    x1_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

    # normalize
    x1_norm = (x1_values - mu[0]) / sigma[0]

    # decision boundary
    x2_norm = -(weights[0] * x1_norm + bias) / weights[1]

    # back to original scale
    x2_values = x2_norm * sigma[1] + mu[1]

    # plot line
    plt.plot(x1_values, x2_values)

    # labels
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")

    plt.show()

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

df["return_1"] = df["Price"].pct_change()



# dataset
X = np.array(
    [
        [2.0, -0.7],
        [1.8, -0.4],
        [1.5, 0.2],
        [1.0, 0.6],
        [0.6, 0.8],
        [0.2, 0.9],
        [0.4, -0.2],
        [1.2, -0.6],
        [0.3, 0.4],
        [1.6, -0.8],
    ],
    dtype=float,
)

# targets 1 → EUR/USD up // 0 → EUR/USD down
y = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 0], dtype=float)

# Z-score normalization
mu = X.mean(axis=0)
sigma = X.std(axis=0)
xn = (X - mu) / sigma

# weights initiation
weights = np.zeros(xn.shape[1])
bias = 0.0

# learning rate
learning_rate = 0.1

# epochs
epochs = 5

for epoch in range(epochs):
    # forward pass (use normalized X)
    z = xn @ weights + bias
    probabilities = sigmoid(z)

    # error
    error = probabilities - y

    # gradients
    dW = (xn.T @ error) / len(xn)
    db = np.mean(error)

    # parameters update (FIXED)
    weights = weights - learning_rate * dW
    bias = bias - learning_rate * db

    # predictions for monitoring
    predictions = (probabilities > 0.5).astype(int)
    accuracy = np.mean(predictions == y.astype(int))

    # calculate loss
    loss = -np.mean(
        y * np.log(probabilities + 1e-9) + (1 - y) * np.log(1 - probabilities + 1e-9)
    )

    print(f"Epoch {epoch}: weights={weights}, bias={bias:.6f}, acc={accuracy:.2f}, loss={loss:.2f}")
    print("predictions:", predictions)

print("\nFinal values:")
print("weights:", weights, "bias:", bias)


