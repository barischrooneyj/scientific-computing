import numpy as np
import matplotlib.pyplot as plt

from methods import jacobi, sor



def makeSink(matrix_len, sinks=[]):
    """Make a sink matrix of given length and sinks at given locations."""
    sink = np.zeros(shape=(matrix_len, matrix_len))
    for (i, j) in sinks:
        sink[i][j] = True
    return sink


def getInitialMatrix(matrix_len):
    """Return the t = 0 matrix with 1 at the top and 0s at the bottom."""
    matrix = np.zeros(shape=(matrix_len, matrix_len))
    matrix[0] = [1] * matrix_len
    matrix[-1] = [0] * matrix_len
    return matrix

def getAnalyticMatrix(matrix_len):
    """Return the t = 0 matrix with 1 at the top and 0s at the bottom."""
    matrix = np.zeros(shape=(matrix_len, matrix_len))
    for i in range(matrix_len):
        matrix[i] = [(matrix_len - 1 - i) / (matrix_len - 1)] * matrix_len
    return matrix


def grow(eta=0.5, omega=1.8, matrix_len=100,
         minimum_c=10**-5, start=None, show=False,
         save=True, load=False, max_sinks = 200):

    """Start at one spot and grow a tree"""
    if start is None:
        start = (matrix_len - 2, int(matrix_len / 2))
        
    fname = "eta-{}-omega-{}-start-{}-min-c-{}-mlen-{}.csv".format(
        eta, omega, start, minimum_c, matrix_len)


    # Load and return previous results if possible.
    matrix = None
    if load:
        try:
            with open(fname) as f:
                print("Loaded matrix from {}".format(fname))
                matrix = np.loadtxt(f)
        except FileNotFoundError:
            print("Could not load matrix from {}".format(fname))

    # If didn't load matrix from file.
    if matrix is None:
        sink = makeSink(matrix_len=matrix_len, sinks=[start])
        print(sink)
        matrix = getAnalyticMatrix(matrix_len)

        for _ in range(max_sinks):
            result = updateMatrix(
                matrix = matrix,
                method=lambda ml, m, t, s: sor(ml, m, t, omega, s),
                sink=sink
            )
            densitymap = growthCandidates(result[0], sink)
            densitymap = [[i, j, c] for [i, j, c] in densitymap if c > minimum_c]
            print("density map after removal = {}".format(densitymap))
            new_sink = newgrowth(eta, densitymap)
            sink[new_sink[0]][new_sink[1]] = True
            matrix = result[0]
    print(matrix)
    print("Num cells = {}".format(np.sum(sink)))
    print(sink)

    # Save the result
    if save:
        with open(fname, "w") as f:
            np.savetxt(f, matrix)
        print("Saved matrix to {}".format(fname))


    #for i in range(matrix_len)):
    #    for j in range(len(matrix[0])):
            
            
    #plt.show()

    if show:
        plt.imshow(matrix)
        plt.show()



def newgrowth(eta, densitymap):
    """Return a new growth candidate."""

    cTotal = 0
    for el in densitymap:
        print("el[2] {}".format(el[2]))
        el[2] = el[2] ** eta
        cTotal += el[2]
    print(cTotal)

    chances = []
    for el in densitymap:
        chances.append(el[2] / cTotal)
    print(np.cumsum(chances))

    number = np.random.random()
    for index, value in enumerate(np.cumsum(chances)):
        if value > number:
            return (densitymap[index][0], densitymap[index][1])

    raise Exception("No new growth candidate selected. chances = {}".format(chances))


def growthCandidates(heatmap, sinks):
    """Return a list of coordinates of new growth candidates."""
    poslist = []
    N = len(heatmap)

    # Look for a candidate at each position.
    for i in range(len(heatmap)):
        for j in range(len(heatmap[0])):
            # Assume initially not a growth candidate.
            possibility = False
            # Becomes a candidate if a sink is a neighbour.
            for di, dj in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                try:
                    if sinks[(i + di) % N][(j + dj) % N]:
                        possibility = True
                        print(i,j)
                except:
                    pass
            # A sink is not a candidate.
            if sinks[i][j]:
                possibility = False
            # Update the list of candidates.
            if possibility:
                poslist.append([i, j, heatmap[i][j]])

    return poslist


def updateMatrix(matrix,  threshold=10 ** -5, sink=None, method=jacobi):
    """Run a simulation until convergence returning final matrix and counter."""
    #print("finalMatrix: N = {} threshold = {} method = {} sink_size = {}".format(
    #    matrix_len, threshold, method, 0 if sink is None else np.sum(sink)))

    terminate = False
    counter = 0
    while not terminate:
        matrix, terminate = method(len(matrix), matrix, threshold, sink)
        counter += 1
        print(counter)

    return matrix, counter


if __name__ == "__main__":

    # Save simulations varying eta.
    etas = list(np.arange(0, 2.1, 0.3))
    print("etas = {}".format(etas))
    for eta in etas:
        grow(eta=eta, load=True, show=True)
