import math
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


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

    # Assume termination until we find a cell above threshold.
    terminate = True

    # For all rows but top and bottom.
    for i in range(1, matrix_len - 1):
        for j in range(matrix_len):

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


def getInitialMatrix(matrix_len):
    """Return the t = 0 matrix with 1 at the top and 0s at the bottom."""
    matrix = np.zeros(shape=(matrix_len, matrix_len))
    matrix[0] = [1] * matrix_len
    matrix[-1] = [0] * matrix_len
    return matrix


def finalMatrix(matrix_len=50, threshold=10 ** -5, sink=None, method=jacobi):
    """Run a simulation until convergence."""
    print("finalMatrix: N = {} threshold = {} method = {} sink_size = {}".format(
        matrix_len, threshold, method, 0 if sink is None else np.sum(sink)))

    matrix = getInitialMatrix(matrix_len)

    terminate = False
    counter = 0
    while (not terminate):
        matrix, terminate = method(matrix_len, matrix, threshold, sink)
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


def makeSink(matrix_len, sinks=[]):
    """Make a sink matrix of given length and sinks at given locations."""
    sink = np.zeros(shape=(matrix_len, matrix_len))
    for (i, j) in sinks:
        sink[i][j] = True
    return sink


color = ["red","green","blue","yellow", "black"]


def plotAnalyticalSolutionsForJacobi(sink=None):
    """Plot the concentration at a height for a number of timesteps."""

    N=100
    a = [i/N for i in range(N+1)]

    plt.figure()

    for i in range(5):
        list = [c(j,1/(10.0**i)) for j in a]
        plt.plot(a, list, c =color[i], label = "Analytic for t = " + str(1/(10.0**i)))

    i=0
    a_sink = makeSink(N, [
        (int(N/2), int(N/2)),
        (int(N/2+1), int(N/2)),
        (int(N/2), int(N/2+1)),
        (int(N/2+1), int(N/2+1))
        ])
    for tv in tvalues(
        matrix_len=N, method=lambda ml, m, t: jacobi(ml, m, t, sink=a_sink)
    )[2]:
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


def findOptimalOmega(matrix_len, method, initial_omega, sink=None):
    """Return the optimal omega using scipy.optimize."""
    def f(omega):
        print("optimize.minimize.f: omega = {}".format(omega))
        result = finalMatrix(
            matrix_len=matrix_len,
            method=lambda ml, m, t, s: method(ml, m, t, omega, s),
            sink=sink
        )
        print("result[1] = {}".format(result[1]))
        return result[1]
    result = optimize.minimize_scalar(f, tol=0.001, bracket=(0.7, 1.7, 1.98))
    print("Optimal result.x = {}".format(result.x))
    return result.x


def makeRectangleSinks(matrix_len, total_size, rec_size=4):
    """Sinks of given total size, each a 4 square rectangle."""

    sink = np.zeros(shape=(matrix_len, matrix_len))
    max_recs = math.ceil(total_size / rec_size)
    max_recs_on_row = math.ceil(math.sqrt(max_recs))
    print("matrix_len = {} total_size = {} rec_size = {} max_recs = {} max_recs_on_row = {}"
          .format(matrix_len, total_size, rec_size, max_recs, max_recs_on_row))

    count = 0
    def countDone(i_, j_):
        """Set sink at given (i, j) and return if we're finished."""
        sink[i_][j_] = True
        nonlocal count
        count += 1
        return count == total_size

    for ith_row in range(max_recs_on_row):
        for jth_col in range(max_recs_on_row):
            margin = 2
            # Calculate the top left coordinate of the rectangle.
            i = int(((matrix_len - 1) - (2 * margin)) * (ith_row / (max_recs_on_row-1 if max_recs_on_row > 1 else 1)) + margin)
            j = int(((matrix_len - 1) - (2 * margin)) * (jth_col / (max_recs_on_row-1 if max_recs_on_row > 1 else 1)) + margin)
            if countDone(i, j): return sink
            if countDone(i, j+1): return sink
            if countDone(i+1, j): return sink
            if countDone(i+1, j+1): return sink


def plotEffectOfSinks(plot_iterations=False, plot_omega=False):
    """Plot the effect of sinks on time to converge and optimal omega."""

    N = 30
    delta_t = 0.001
    final_time = 1
    default_omega = 1.8

    # A list of sinks of increasing size.
    sinks = [makeRectangleSinks(N, size) for size in range(1, 200+1, 2)]
    print(sinks[-1])
    sink_sizes = [np.sum(sink) for sink in sinks]

    # These will be filled in by the simulations.
    iterations = []
    optimal_omegas = []

    # For each sink sun simulation and record results.
    for i, sink in enumerate(sinks):

        if plot_iterations:
            print("\nFinding iterations")
            matrix, simulation_iterations = finalMatrix(
                matrix_len=N,
                threshold=2 * np.finfo(np.float32).eps,
                sink=sink,
                method=lambda ml, m, t, s: sor(ml, m, t, default_omega, s)
            )
            iterations.append(simulation_iterations)

        if plot_omega:
            print("\nFinding optimal omega")
            optimal_omegas.append(findOptimalOmega(
                matrix_len=N,
                initial_omega=default_omega,
                sink=sink,
                method=sor
            ))
            print("Sink = {} size = {}".format(i, sink_sizes[i]))
            if plot_iterations: print("iterations = {}".format(iterations))
            if plot_omega: print("optimal omegas = {}".format(optimal_omegas))

    print(sink_sizes)
    print(iterations)
    print(optimal_omegas)

    if plot_iterations:
        plt.plot(sink_sizes, iterations)
        plt.title("Timesteps as a function of sink size")
        plt.show()
        plt.close()

    if plot_omega:
        plt.plot(sink_sizes, optimal_omegas)
        plt.title("Optimal omega as a function of sink size")
        plt.xlabel("Number of sinks")
        plt.ylabel("Optimal omega")
        plt.show()


def imshowSinks(matrix_len, amount_sinks):
    """Imshow matrices with sinks."""
    plt.imshow(makeRectangleSinks(matrix_len, amount_sinks))
    plt.show()


if __name__ == "__main__":
    # plotTimeToConverge()
    # plotAnalyticalSolutionsForJacobi()
    # plotTValues()

    if sys.argv[1] == "J":
        plotEffectOfSinks(plot_iterations=True, plot_omega=False)
        imshowSinks(30, 36)
        imshowSinks(30, 37)

