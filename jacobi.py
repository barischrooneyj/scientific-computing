import numpy as np
import matplotlib.pyplot as plt

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

            prev_value = matrix[i][j]
            matrix[i][j] = 0.25 * (
                matrix[i + 1][j]
                + matrix[i - 1][j]
                + matrix[i][(j + 1) % matrix_len]
                + matrix[i][(j - 1) % matrix_len]
            )

            if abs(matrix[i][j] - prev_value) > threshold:
                terminate = False

    return (matrix, terminate)


def sor(matrix_len, matrix, threshold, omega):
    """A new matrix based on matrix at previous time, successive over relaxation."""

    terminate = True

    # For all rows but top and bottom.
    for i in range(1, matrix_len - 1):
        for j in range(matrix_len):

            prev_value = matrix[i][j]
            matrix[i][j] = (omega * 0.25 * (
                matrix[i + 1][j]
                + matrix[i - 1][j]
                + matrix[i][(j + 1) % matrix_len]
                + matrix[i][(j - 1) % matrix_len]
            )) + ((1 - omega) * prev_value)
            #print((prev_value, matrix[i][j]))

            if abs(matrix[i][j] - prev_value) > threshold:
                terminate = False

    return (matrix, terminate)


def getInitialMatrix(matrix_len):
    """Return the t = 0 matrix with 1 at the top and 0s at the bottom."""
    matrix = np.zeros(shape=(matrix_len, matrix_len))
    matrix[0] = [1] * matrix_len
    matrix[-1] = [0] * matrix_len
    return matrix


def finalMatrix(matrix_len=50, threshold=10 ** -5, method=jacobi):
    """Perform Jacobi iteration until convergence."""

    matrix = getInitialMatrix(matrix_len)

    terminate = False
    counter = 0
    while (not terminate):
        matrix, terminate = method(matrix_len, matrix, threshold)
        counter += 1

    return matrix

def tvalues(matrix_len=50, threshold=10 ** -5, method=jacobi):

    dot = []
    """Perform Jacobi iteration until convergence."""

    matrix = getInitialMatrix(matrix_len)

    terminate = False
    counter = 0
    while (not terminate):
        matrix, terminate = method(matrix_len, matrix, threshold)
        counter += 1
        # print(counter)

    t = [round(10**-i * counter) for i in range(5)]+[counter - 1]

    matrix = getInitialMatrix(matrix_len)

    terminate = False
    counter2 = 0
    while (not terminate):
        matrix, terminate = method(matrix_len, matrix, threshold)

        if counter2 in t:
            temparray = []
            for row in matrix:
                temparray.append(row[0])
            dot.append(temparray)
        counter2 += 1

    return matrix, counter, dot


print(finalMatrix(method=jacobi))
# print(finalMatrix(method=gaussSeidel))
for list in tvalues(method=lambda ml, m, t: sor(ml, m, t, 1.9))[2]:
    list.reverse()
    plt.plot(range(len(list)), list)
plt.show()
