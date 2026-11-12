import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    
    A = np.array(A)
    T = np.array(T)
    S = np.array(S)

    T_det = np.linalg.det(T)
    S_det = np.linalg.det(S)

    if T_det != 0 and S_det != 0:

        T_inv = np.linalg.inv(T) 
        
        transformed_matrix =  T_inv @ A
        transformed_matrix = transformed_matrix @ S
    else:
        transformed_matrix = -1
    
    return transformed_matrix