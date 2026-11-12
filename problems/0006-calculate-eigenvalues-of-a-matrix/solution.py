def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:

    # det(A-lambda*I) = 0
    # 
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    trace = matrix[0][0] + matrix[1][1]

    # x*x - trace * x + det = 0
    # x = trace +- sqrt(trace**2 - 4 * det)

    eigenvalues = [(trace + (trace**2 - 4 * det) ** 0.5)/2, (trace - (trace**2 - 4 * det) ** 0.5)/2]



    return eigenvalues