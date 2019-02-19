import numpy as np
import matplotlib.pyplot as plt
import math

# 1.3


def jacobi(matrix_len, prev_matrix, threshold):
	"""A new matrix based on matrix at previous time."""
	
	sink = np.zeros(shape=(matrix_len, matrix_len))
	#sink[2][2] = True
	
	new_matrix = np.zeros(shape=(matrix_len, matrix_len))
	
	# Top and bottom rows.
	new_matrix[0] = [1] * matrix_len
	new_matrix[-1] = [0] * matrix_len
	
	terminate = True
	
	# For all rows but top and bottom.
	for i in range(1, matrix_len - 1):
		for j in range(matrix_len):
	
			if not sink[i][j]:
				new_matrix[i][j] = 0.25 * (
					prev_matrix[i + 1][j]
					+ prev_matrix[i - 1][j]
					+ prev_matrix[i][(j + 1) % matrix_len]
					+ prev_matrix[i][(j - 1) % matrix_len]
				)
			else: 
				new_matrix[i][j] = 0
				
	
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


def finalMatrix(matrix_len=25, threshold=10 ** -5, method=jacobi):
	"""Perform Jacobi iteration until convergence."""
	
	# Set up initial matrix.
	matrix = np.zeros(shape=(matrix_len, matrix_len))
	matrix[0] = [1] * matrix_len
	matrix[-1] = [0] * matrix_len
	
	terminate = False
	counter = 0
	while (not terminate):
		matrix, terminate = method(matrix_len, matrix, threshold)
		counter += 1
		
		
	return matrix

def tvalues(matrix_len=50, threshold=10 ** -5, method=jacobi):

	dot = []
	"""Perform Jacobi iteration until convergence."""
	
	# Set up initial matrix.
	matrix = np.zeros(shape=(matrix_len, matrix_len))
	matrix[0] = [1] * matrix_len
	matrix[-1] = [0] * matrix_len
	
	terminate = False
	counter = 0
	while (not terminate):
		matrix, terminate = method(matrix_len, matrix, threshold)
		counter += 1

 
	return matrix, counter, dot

# SOR algorithms with different omega's
def sorWith3Args(matrix_len, matrix, threshold):
	return sor(matrix_len, matrix, threshold, 1.9)
	
def sorWith3Args2(matrix_len, matrix, threshold):
	return sor(matrix_len, matrix, threshold, 1.5)
	
def sorWith3Args3(matrix_len, matrix, threshold):
	return sor(matrix_len, matrix, threshold, 0.5)
	

thresholds = [10**-i for i in range(1,7)]

jacobiitt = []
gaussSeidelitt = []
sor19itt = []
sor15itt = []
sor05itt = []


for threshold in thresholds:
	jacobiitt.append(tvalues(threshold = threshold)[1])
	gaussSeidelitt.append(tvalues(threshold = threshold, method = gaussSeidel)[1])
	sor19itt.append(tvalues(threshold = threshold, method = sorWith3Args)[1])
	sor15itt.append(tvalues(threshold = threshold, method = sorWith3Args2)[1])
	sor05itt.append(tvalues(threshold = threshold, method = sorWith3Args3)[1])
	
	print(threshold, jacobiitt[-1], gaussSeidelitt[-1], sor15itt[-1], sor19itt[-1])
	
plt.plot(range(1,7), jacobiitt, label ="Jacobi")
plt.plot(range(1,7), gaussSeidelitt, label = "GaussSeidel")
plt.plot(range(1,7), sor19itt, label = "SOR with ω = 1.9")
plt.plot(range(1,7), sor15itt, label = "SOR with ω = 1.5")
plt.plot(range(1,7), sor05itt, label = "SOR with ω = 0.5")
plt.yscale('log')
plt.xlabel("δt = 10^i")
plt.ylabel("Iterations")
plt.legend()
plt.show()

