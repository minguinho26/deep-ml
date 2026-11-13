import numpy as np
import re

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
    
    Returns:
        Jacobian matrix as list of lists
    """
    x = np.asarray(x, dtype=float)
    fx = np.asarray(f(x), dtype=float)
    J = np.zeros((fx.size, x.size))

    for j in range(x.size):
        # 한쪽 차원만 편미분 하기 때문에 한 쪽 차원만 h를 설정
        e = np.zeros_like(x) 
        e[j] = h

        J[:, j] = (np.asarray(f(x + e)) - np.asarray(f(x))) / h


    return J.tolist()