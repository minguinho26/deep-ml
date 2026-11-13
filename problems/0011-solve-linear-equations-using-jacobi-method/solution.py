import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:

    if len(b.shape) == 1:
        x = np.zeros([A.shape[1]])
    else:
        x = np.zeros([A.shape[1], b.shape[1]])

    for i in range(n):

        new_x = x.copy()

        for i in range(x.shape[0]):
            new_x[i] = (1/A[i,i]) * (b[i] - sum([A[i,j] * x[j] for j in range(A.shape[1]) if i != j]))

        x = new_x 

    return x