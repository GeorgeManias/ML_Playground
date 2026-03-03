# Logistic Regression – One Step

This project shows how Logistic Regression works using NumPy.

It performs:
- Forward calculation (z = XW + b)
- Sigmoid transformation
- Prediction
- One step of Gradient Descent

---

## What the script does

1. Defines a small dataset
2. Sets initial weights and bias
3. Calculates accuracy before training
4. Computes error
5. Updates weights and bias (1 step)
6. Shows new accuracy

---

## Math Used

z = X · W + b  

sigmoid(z) = 1 / (1 + e^(-z))  

dW = (Xᵀ · error) / n  

db = mean(error)  

W = W - lr · dW  

b = b - lr · db  

---

## Run