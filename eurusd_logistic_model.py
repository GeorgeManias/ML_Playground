import numpy as np

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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

    print(f"Epoch {epoch}: weights={weights}, bias={bias:.6f}, acc={accuracy:.2f}")
    print("predictions:", predictions)

print("\nFinal values:")
print("weights:", weights, "bias:", bias)
