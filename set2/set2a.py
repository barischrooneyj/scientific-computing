import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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


def grow(eta=0.5, omega=1.8, matrix_len=100, stop_at_top=True,
         minimum_c=10**-5, start=None, show=False, fname_append="",
         save=True, load=False, max_sinks = 200):
    """Start at one spot and grow a tree.

    Optionally load or save images and data to file.
    Optionally show images.

    Returns the amount of cells grown after the initial cell.
    """
    if start is None:
        start = (matrix_len - 1, int(matrix_len / 2))

    fname = "2a-eta-{}-omega-{}-start-{}-min-c-{}-mlen-{}{}.csv".format(
        np.around(eta, 4), omega, start, minimum_c, matrix_len, fname_append)

    # Load and return previous results if possible.
    matrix = None
    sink = None
    loaded = False
    if load:
        try:
            with open("simulations/" + fname) as f:
                matrix = np.loadtxt(f)
            with open("simulations/" + fname + ".sink") as f:
                sink = np.loadtxt(f)
            print("Loaded concentrations and sinks from \nsimulations/{}(.sink)".format(fname))
            loaded = True
        except FileNotFoundError:
            print("Could not load concentrations and sinks from \nsimulations/{}(.sink)".format(fname))

    # If didn't load matrix from file.
    if matrix is None:
        matrix = getAnalyticMatrix(matrix_len)
        sink = makeSink(matrix_len=matrix_len, sinks=[start])

        for t in range(max_sinks):
            print("Running SOR at t = {}".format(t))
            result = updateMatrix(
                matrix = matrix,
                method=lambda ml, m, t, s: sor(ml, m, t, omega, s),
                sink=sink
            )
            print("Looking for growth candidate at t = {}".format(t))
            densitymap = growthCandidates(result[0], sink)
            densitymap = [[i, j, c] for [i, j, c] in densitymap if c > minimum_c]
            new_sink = newgrowth(eta, densitymap)
            sink[new_sink[0]][new_sink[1]] = True
            # Stop once the top is reached.
            if stop_at_top and new_sink[0] == 0:
                break
            matrix = result[0]

    # Save the result.
    if not loaded and save:
        with open("simulations/" + fname, "w") as f:
            np.savetxt(f, matrix)
        with open("simulations/" + fname + ".sink", "w") as f:
            np.savetxt(f, sink)
        print("Saved concentrations and sinks to \nsimulations/{}(.sink)".format(
            fname))

    if save or show:
        # Set sinks to a negative value for greater colour contrast.
        for i in range(matrix_len):
            for j in range(matrix_len):
                if sink[i][j]:
                    matrix[i][j] = -1

        masked_array = np.ma.masked_where(matrix == -1, matrix)

        cmap = matplotlib.cm.coolwarm
        cmap.set_bad(color="white")

        plt.imshow(masked_array, cmap=cmap)
        plt.xticks([x for x in range(0, matrix_len, 10)] + [matrix_len - 1])
        plt.yticks([x for x in range(0, matrix_len, 10)] + [matrix_len - 1])
        plt.xlabel("Spatial dimension x")
        plt.ylabel("Spatial dimension y")
        plt.title("DLA, η = {:.1f}, cells grown = {}".format(eta, int(np.sum(sink) - 1)))

    if show:
        plt.show()
    if save:
        plt.savefig("images/{}.png".format(fname[:-4]))
        print("Saved image to\nimages/{}.png".format(fname))

    return np.sum(sink) - 1


def newgrowth(eta, densitymap):
    """Return a new growth candidate.

    densitymap is a list of (i, j, c) coordinates and concentration c.
    """

    cTotal = 0
    for el in densitymap:
        el[2] = el[2] ** eta
        cTotal += el[2]

    chances = []
    for el in densitymap:
        chances.append(el[2] / cTotal)

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
                    sink_i = i + di
                    sink_j = (j + dj) % N
                    # Is sink is in bounds and sink is a neighbour.
                    if 0 <= sink_i and sink_i < N and sinks[sink_i][sink_j]:
                        possibility = True
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

    terminate = False
    counter = 0
    while not terminate:
        matrix, terminate = method(len(matrix), matrix, threshold, sink)
        counter += 1

    return matrix, counter


def plotImpactOfEta(matrix_len=100, start=1.2, stop=12, step=0.3, repeat=3):
    """Run a growth simulation and plot eta against growth height/width dimensions."""
    etas = np.arange(start, stop + (step * 0.1), step)
    # For each eta a list of cells grown.
    cells_grown = np.zeros(shape=(len(etas), repeat))
    for e, eta in enumerate(etas):
        eta = np.around(eta, 4)
        print("eta = {}".format(eta))
        for r in range(repeat):
            print("Running repeat {} for eta {}".format(r, eta))
            cells_grown[e][r] =  grow(
                eta=eta, matrix_len=matrix_len, max_sinks=matrix_len ** 2,
                load=True, save=False, show=False,
                fname_append="" if r == 0 else "-{}".format(r)
            )
    plt.close()
    plt.plot(etas, [np.mean(g / matrix_len) for g in cells_grown], label="mean")
    plt.plot(etas, [np.min(g / matrix_len) for g in cells_grown], label="min")
    plt.plot(etas, [np.max(g / matrix_len) for g in cells_grown], label="max")
    plt.legend()
    plt.title("Growths per container height, varying η")
    plt.xlabel("η")
    plt.ylabel("Growths when top reached / container height")
    fname = "images/2a-eta-impact-start-{}-stop-{}-step-{}-repeat-{}-N-{}.png".format(
        start, stop, step, repeat, matrix_len)
    plt.savefig(fname)
    print("Saved image to {}".format(fname))
    plt.show()


def plotGrowths(matrix_len=100, start=1.2, stop=2, step=0.3):
    """Plot the growth of DLA for various values of eta."""
    etas = list(np.arange(0, 12.1, 0.3))
    print("etas = {}".format(etas))
    for eta in etas:
        print("eta = {}".format(np.around(eta, 4)))
        grow(eta=eta, matrix_len=100, max_sinks=1000, load=True, show=True)


if __name__ == "__main__":

    # plotGrowths()
    plotImpactOfEta(start=4.2)
