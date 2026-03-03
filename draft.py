import numpy as np


def sigmoid(z):
    # TODO
    pass


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

y = np.array([
    # TODO targets
], dtype=float)


# normalization
mu = X.mean(axis=0)
sigma = X.std(axis=0)

# TODO
Xn = ...


# parameters
weights = np.zeros(...)
bias = ...


learning_rate = ...
epochs = ...


for epoch in range(...):

    # forward pass
    z = ...

    probabilities = ...

    # loss (optional)
    # TODO

    # error
    error = ...

    # gradients
    dW = ...
    db = ...

    # update
    weights = ...
    bias = ...


    if epoch % 200 == 0:

        predictions = ...

        accuracy = ...

        print("Epoch:", epoch, "Accuracy:", accuracy)


print("Final weights:", weights)
print("Final bias:", bias)