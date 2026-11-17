import numpy as np

# 코드가 문제있나? 
# Use full-batch gradient descent: in each epoch, perform a single forward pass over all training examples, compute the MSE loss and its gradients averaged across the entire batch, then apply one weight and bias update per epoch. Record the MSE before each update (i.e., using the weights at the start of the epoch), so mse_values has length equal to epochs. Return the updated weights, bias, and the list of MSE values, each rounded to four decimal places.
# Use full-batch gradient descent: in each epoch, perform a single forward pass over all training examples, compute the MSE loss and its gradients averaged across the entire batch, then apply one weight and bias update per epoch. Record the MSE before each update (i.e., using the weights at the start of the epoch), so mse_values has length equal to epochs. Return the updated weights, bias, and the list of MSE values, each rounded to four decimal places.
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    # Your code here
    updated_weights = initial_weights.copy()
    updated_bias = initial_bias
    n = len(features)
    mse_values = []

    for epoch in range(epochs):
        
        z = features @ updated_weights + updated_bias
        predictions = 1 / (1 + np.exp(-z))
        
        errors = predictions - labels
        mse = np.mean(errors ** 2)
        mse_values.append(round(float(mse), 4))

        delta = 2 * errors * predictions * (1 - predictions) # signoid 고려해서 미분
        
        gradient_weights = (features.T @ delta) / n
        gradient_bias = np.sum(delta) / n

        updated_weights -= learning_rate * gradient_weights
        updated_bias -= learning_rate * gradient_bias

    return np.round(updated_weights, 4).tolist(), np.round(updated_bias, 4).item(), np.round(mse_values, 4).tolist()