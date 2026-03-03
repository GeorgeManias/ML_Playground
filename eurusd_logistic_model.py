import numpy as np

#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# dataset
X = np.array([
    # diff_rate , risk
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
], dtype=float)

#targets 1 → EUR/USD up//0 → EUR/USD down
y = np.array([
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    0
], dtype=float)


#Z-score normalization
mu=X.mean(axis=0)
sigma=X.std(axis=0)
xn=(X-mu)/sigma

#weights_initiation
weights=np.zeros(X.shape[1])
bias = 0.0

