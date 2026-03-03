import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Data (X) - use float for safer math with gradients
X = np.array([
    [1, 6],
    [2, 5],
    [8, 9],
    [2, 3]
], dtype=float)

# Weights MUST be float (if you use dtype=int, 0.8 becomes 0!)
weights = np.array([1.2, 0.8], dtype=float)
bias = 0.9

# Labels (targets)
y = np.array([1, 1, 0, 0], dtype=float)

# Calculations Before Training
z = X @ weights + bias
probabilities = sigmoid(z)
predictions = (probabilities > 0.5).astype(int)
accuracy = np.mean(predictions == y.astype(int))

print("--Values Before--")
print("Accuracy:", accuracy)
print("Weights:", weights, "Bias:", bias)

print("--One Round Learning--")
learning_rate = 0.1

# error
error = probabilities - y

# gradients
dW = (X.T @ error) / len(X)
db = np.mean(error)

# update parameters (FIXED)
weights = weights - learning_rate * dW
bias = bias - learning_rate * db

# Re-check after 1 step
z = X @ weights + bias
probabilities = sigmoid(z)
predictions = (probabilities > 0.5).astype(int)
new_accuracy = np.mean(predictions == y.astype(int))

print("\nAfter 1 step - accuracy:", new_accuracy)
print("After 1 step - weights:", weights, "bias:", bias)
print("After 1 step - predictions:", predictions)
print("Targets (y):", y.astype(int))