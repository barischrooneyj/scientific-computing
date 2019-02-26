import numpy as np

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


def grow(eta=0.5, omega=1.8, start=(8, 9),
         minimum_c=10**-5, matrix_len=100):
    """Start at one spot and grow a tree"""

    sink = makeSink(matrix_len=matrix_len, sinks=[start])
    matrix = getInitialMatrix(matrix_len)

    for _ in range(70):
        result = updateMatrix(
            matrix = matrix,
            method=lambda ml, m, t, s: sor(ml, m, t, omega, s),
            sink=sink
        )
        densitymap = makePossibilties(result[0], sink)
        densitymap = [[i, j, c] for [i, j, c] in densitymap if c > minimum_c]
        print("density map after removal = {}".format(densitymap))
        new_sink = newgrowth(eta, densitymap)
        sink[new_sink[0]][new_sink[1]] = True
        matrix = result[0]

    print(matrix)
    print("Num cells = {}".format(np.sum(sink)))


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
    print(heatmap, sinks)
    poslist = []
    N = len(heatmap)
    print("len(heatmap) = {}".format(N))

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
    """Run a simulation until convergence."""
    #print("finalMatrix: N = {} threshold = {} method = {} sink_size = {}".format(
    #    matrix_len, threshold, method, 0 if sink is None else np.sum(sink)))

    terminate = False
    counter = 0
    while (not terminate):
        matrix, terminate = method(len(matrix), matrix, threshold, sink)
        counter += 1
        print(counter)

    return matrix, counter


if __name__ == "__main__":
    grow()
