import numpy as np
import matplotlib.pyplot as plt
import math

# 1.3


def jacobi(matrix_len, prev_matrix, threshold):
	
	
	sink = makeSink(matrix_len)
	"""A new matrix based on matrix at previous time."""
	
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
	
	
	sink = makeSink(matrix_len)
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
	
	sink = makeSink(matrix_len)
	
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

def tvalues(matrix_len=30, threshold=10 ** -5, method=jacobi):

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
		print(counter)
		
	t = [round(10**-i * counter) for i in range(5)]
	
	# Set up initial matrix.
	matrix = np.zeros(shape=(matrix_len, matrix_len))
	matrix[0] = [1] * matrix_len
	matrix[-1] = [0] * matrix_len
	
	terminate = False
	counter2 = 0
	while (not terminate):
		matrix, terminate = method(matrix_len, matrix, threshold)
		
		if counter2 in t:
			t.remove(counter2)
			temparray = []
			for row in matrix:
				temparray.append(row[0])
			dot.append(temparray)
		counter2 += 1
		
	temparray = []
	for row in matrix:
		temparray.append(row[0])
 
	return matrix, counter, dot


def sorWith3Args(matrix_len, matrix, threshold):
	return sor(matrix_len, matrix, threshold, 1.9)
	
def c(x,t):
	D = 1
	answer = 0
	for i in range(100):
		newit = math.erfc((1-x+2*i)/(2*math.sqrt(D*t))) - math.erfc((1+x+2*i)/(2*math.sqrt(D*t)))
		answer += newit
	return answer
	
def makeSink(matrix_len):
	sink = np.zeros(shape=(matrix_len, matrix_len))
	sink[int(matrix_len/2)][int(matrix_len/2)] = True
	sink[int(matrix_len/2+1)][int(matrix_len/2)] = True
	sink[int(matrix_len/2)][int(matrix_len/2+1)] = True
	sink[int(matrix_len/2+1)][int(matrix_len/2+1)] = True
	return sink
	

N=100
a = [i/N for i in range(N+1)]

plt.figure()

color = ["red","green","blue","yellow", "black"]

for i in range(5):
	list = [c(j,1/(10.0**i)) for j in a]
	plt.plot(a, list, c =color[i], label = "Analytic for t = " + str(1/(10.0**i)))
	
i=0
for tv in tvalues(method=jacobi)[2]:
	tv.reverse()
	plt.plot([el/(len(tv)-1) for el in range(len(tv))] , tv, c = color[-i-1], linestyle= ":", label="Jacobi")
	i += 1
	
#i = 0
#tvall = tvalues(method=sorWith3Args)[2]
#for tv in tvall:
#	tv.reverse()
#	plt.plot([el/(len(tv)-1) for el in range(len(tv))], tv, c = color[-i-1], linestyle= "-.", label="SOR for t = " + str(1/(10.0**(len(tvall)-i))))
#	i += 1

#i = 0	
#for tv in tvalues(method=gaussSeidel)[2]:
#	tv.reverse()
#	plt.plot([el/(len(tv)-1) for el in range(len(tv))], [tvi/max(tv) for tvi in tv], c = color[i], linestyle= "--", label="Jacobi")
#	i += 1
	
plt.legend()
plt.show()
#print(finalMatrix(method=gaussSeidel))
#for list in tvalues(method=sorWith3Args)[2]:
#	list.reverse()
#	plt.plot(range(len(list)), list)
#plt.show()




