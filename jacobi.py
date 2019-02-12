import numpy as np

# 1.3


def jacobi(matrix_len, prev_matrix, threshold):
    """A new matrix based on matrix at previous time."""

    new_matrix = np.zeros(shape=(matrix_len, matrix_len))

    # Top and bottom rows.
    new_matrix[0] = [1] * matrix_len
    new_matrix[-1] = [0] * matrix_len

    terminate = True

    # For all rows but top and bottom.
    for i in range(1, matrix_len - 1):
        for j in range(matrix_len):

            new_matrix[i][j] = 0.25 * (
                prev_matrix[i + 1][j]
                + prev_matrix[i - 1][j]
                + prev_matrix[i][(j + 1) % matrix_len]
                + prev_matrix[i][(j - 1) % matrix_len]
            )

            if abs(prev_matrix[i][j] - new_matrix[i][j]) > threshold:
                terminate = False

    return (new_matrix, terminate)


def gaussSeidel(matrix_len, matrix, threshold):
    """A new matrix based on matrix at previous time, updated in place."""

    terminate = True

    # For all rows but top and bottom.
    for i in range(1, matrix_len - 1):
        for j in range(matrix_len):

            prev_value = matrix[i][i]
            matrix[i][j] = 0.25 * (
                matrix[i + 1][j]
                + matrix[i - 1][j]
                + matrix[i][(j + 1) % matrix_len]
                + matrix[i][(j - 1) % matrix_len]
            )

            if abs(matrix[i][j] - prev_value) > threshold:
                terminate = False

    return (matrix, terminate)


def finalMatrix(matrix_len=6, time_steps=100, threshold=10 ** -5, method=jacobi):
    """Perform Jacobi iteration until convergence."""

    # Set up initial matrix.
    matrix = np.zeros(shape=(matrix_len, matrix_len))
    matrix[0] = [1] * matrix_len
    matrix[-1] = [0] * matrix_len

    terminate = False
    while (not terminate):
        matrix, terminate = method(matrix_len, matrix, threshold)
    return matrix


print(finalMatrix(method=gaussSeidel))
