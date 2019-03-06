import numpy as np
import numba


def jacobi(matrix_len, prev_matrix, threshold, sink=None):
    """A new matrix based on matrix at previous time.

    Supports an optional sink argument as a matrix of dimensions (matrix_len,
    matrix_len).

    """

    new_matrix = np.zeros(shape=(matrix_len, matrix_len))

    # Top and bottom rows.
    new_matrix[0] = [1] * matrix_len
    new_matrix[-1] = [0] * matrix_len

    terminate = True

    # For all rows but top and bottom.
    for i in range(1, matrix_len - 1):
        for j in range(matrix_len):

            if sink is not None and not sink[i][j]:
                new_matrix[i][j] = 0.25 * (
                    prev_matrix[i + 1][j]
                    + prev_matrix[i - 1][j]
                    + prev_matrix[i][(j + 1) % matrix_len]
                    + prev_matrix[i][(j - 1) % matrix_len]
                )
            else:
                prev_value = matrix[i][j]
                new_matrix[i][j] = 0

            if abs(prev_matrix[i][j] - new_matrix[i][j]) > threshold:
                terminate = False

    return (new_matrix, terminate)


def gaussSeidel(matrix_len, matrix, threshold, sink=None):
    """A new matrix based on matrix at previous time, updated in place.

    Supports an optional sink argument as a matrix of dimensions (matrix_len,
    matrix_len).

    """

    terminate = True

    # For all rows but top and bottom.
    for i in range(1, matrix_len - 1):
        for j in range(matrix_len):

            prev_value = matrix[i][j]
            if sink is None:
                matrix[i][j] = 0.25 * (
                    matrix[i + 1][j]
                    + matrix[i - 1][j]
                    + matrix[i][(j + 1) % matrix_len]
                    + matrix[i][(j - 1) % matrix_len]
                )
            else:
                if sink[i][j]:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = 0.25 * (
                    matrix[i + 1][j]
                    + matrix[i - 1][j]
                    + matrix[i][(j + 1) % matrix_len]
                    + matrix[i][(j - 1) % matrix_len]
                )

            #print(matrix[i][j], prev_value)
            if abs(matrix[i][j] - prev_value) > threshold:
                terminate = False

    return (matrix, terminate)


@numba.jit(nopython=True, parallel=True)
def sor(matrix_len, matrix, threshold, omega, sink=None):
    """A new matrix based on matrix at previous time, successive over relaxation.

    Supports an optional sink argument as a matrix of dimensions (matrix_len,
    matrix_len).

    """
    start_i = 1
    start_j = 0
    end_i = matrix_len - 2
    end_j = matrix_len - 1

    # Assume termination until we find a cell above threshold.
    terminate = True

    # For all rows but top and bottom.
    for i in range(start_i, end_i + 1):
        for j in range(start_j, end_j + 1):

            prev_value = matrix[i][j]
            if sink is None or not sink[i][j]:
                matrix[i][j] = (omega * 0.25 * (
                    matrix[i + 1][j]
                    + matrix[i - 1][j]
                    + matrix[i][(j + 1) % matrix_len]
                    + matrix[i][(j - 1) % matrix_len]
                )) + ((1 - omega) * prev_value)
            else:
                matrix[i][j] = 0

            if abs(matrix[i][j] - prev_value) > threshold:
                terminate = False

    return (matrix, terminate)
