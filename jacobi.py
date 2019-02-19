import math

import numpy as np
import matplotlib.pyplot as plt


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
            if sink is not None and not sink[i][j]:
                matrix[i][j] = 0.25 * (
                    matrix[i + 1][j]
                    + matrix[i - 1][j]
                    + matrix[i][(j + 1) % matrix_len]
                    + matrix[i][(j - 1) % matrix_len]
                )
            else:
                new_matrix[i][j] = 0

            if abs(matrix[i][j] - prev_value) > threshold:
                terminate = False

    return (matrix, terminate)


def sor(matrix_len, matrix, threshold, omega, sink=None):
    """A new matrix based on matrix at previous time, successive over relaxation.

    Supports an optional sink argument as a matrix of dimensions (matrix_len,
    matrix_len).

    """

    terminate = True

    # For all rows but top and bottom.
    for i in range(1, matrix_len - 1):
        for j in range(matrix_len):

            prev_value = matrix[i][j]
            if sink is not None and not sink[i][j]:
                matrix[i][j] = (omega * 0.25 * (
                    matrix[i + 1][j]
                    + matrix[i - 1][j]
                    + matrix[i][(j + 1) % matrix_len]
                    + matrix[i][(j - 1) % matrix_len]
                )) + ((1 - omega) * prev_value)
            else:
                new_matrix[i][j] = 0

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

    return matrix, counter


def tvalues(matrix_len=50, threshold=10 ** -5, method=jacobi):
    """Perform Jacobi iteration until convergence."""

    dot = []
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


def plotConvergenceOverNAndOmega(filename, omegas, Ns):
    """A plot of time to converge for multiple lines (N) against omega."""
    for N in Ns:
        times = []
        for omega in omegas:
            _, time = finalMatrix(
                matrix_len=N, method=lambda ml, m, t: sor(ml, m, t, omega))
            times.append(time)
        print("N = {} min_time = {} omega_min_time = {}".format(
            N, min(times), omegas[times.index(min(times))]))
        plt.plot(omegas, times, label="N = {}".format(N))
    plt.title("Timesteps to converge as a function of N and ω")
    plt.xlabel("ω")
    plt.ylabel("Timesteps to converge")
    plt.legend()
    plt.savefig(filename)
    plt.show()
    print("Saved {}".format(filename))


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


color = ["red","green","blue","yellow", "black"]


def plotAnalyticalSolutionsForJacobi():
    """Plot the concentration at a height for a number of timesteps."""

    N=100
    a = [i/N for i in range(N+1)]

    plt.figure()

    for i in range(5):
        list = [c(j,1/(10.0**i)) for j in a]
        plt.plot(a, list, c =color[i], label = "Analytic for t = " + str(1/(10.0**i)))

    i=0
    for tv in tvalues(method=jacobi)[2]:
        tv.reverse()
        plt.plot([el/(len(tv)-1) for el in range(len(tv))] , tv, c = color[-i-1], linestyle= ":", label="Jacobi")
        i += 1

    plt.legend()
    plt.show()


def plotTValues():

    omega = 1.9
    i = 0

    tvall = tvalues(method=lambda ml, m, t: sor(ml, m, t, omega))[2]
    for tv in tvall:
    	tv.reverse()
    	plt.plot([el/(len(tv)-1) for el in range(len(tv))], tv, c = color[-i-1], linestyle= "-.", label="SOR for t = " + str(1/(10.0**(len(tvall)-i))))
    	i += 1

    i = 0
    for tv in tvalues(method=gaussSeidel)[2]:
    	tv.reverse()
    	plt.plot([el/(len(tv)-1) for el in range(len(tv))], [tvi/max(tv) for tvi in tv], c = color[i], linestyle= "--", label="Jacobi")
    	i += 1

    print(finalMatrix(method=gaussSeidel))
    for list in tvalues(method=lambda ml, m, t: sor(ml, m, t, omega))[2]:
    	list.reverse()
    	plt.plot(range(len(list)), list)

    plt.show()


def plotTimeToConverge():
    """Plot time to converge over N and omega and find optimal omega."""

    Ns_and_ranges = [
        (10, 1.62-.05, 1.62+.05),
        (15, 1.73-.05, 1.73+.05),
        (20, 1.80-.05, 1.80+.05),
        (25, 1.84-.05, 1.84+.05),
        (30, 1.87-.05, 1.87+.05),
        (35, 1.88-.05, 1.88+.05),
        (40, 1.90-.05, 1.90+.05)
    ]

    # Plot for all N.
    plotConvergenceOverNAndOmega(
        "1-J-optimal-omega.png",
        omegas=np.linspace(1.6, 1.95, 100),
        Ns=[N for N, _1, _2 in Ns_and_ranges]
    )

    # Find optimal omega per N.
    for N, range_min, range_max in Ns_and_ranges:
        plotConvergenceOverNAndOmega(
            "_unused.png",
            omegas=np.linspace(range_min, range_max, 100),
            Ns=[N]
        )


if __name__ == "__main__":
    # plotTimeToConverge()
    plotAnalyticalSolutionsForJacobi()
    # plotTValues()
