import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here

    # standardize the dataset
    standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # compute the covariance matrix
    cov = np.cov(standardized_data, rowvar=False)

    # find the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # return the principal components (the eigenvectors corresponding to the largest eigenvalues).
    principal_components = eigenvectors[:, np.argsort(eigenvalues)[::-1][:k]]

    # find its first element with absolute value > 1e-10; if that element is negative, multiply the entire eigenvector by -1.
    for i in range(principal_components.shape[1]):
        if principal_components[0, i] < 0:
            principal_components[:, i] *= -1

    return principal_components
