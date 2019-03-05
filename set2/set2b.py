import numpy as np
import matplotlib.pyplot as plt

# Top row i = 0.
# Left column j = 0.

DEBUG = False


def p(s):
    """Print if DEBUG is true."""
    if DEBUG:
        print(s)


def initialWalkerMatrix(matrix_len, initial_walker):
    """An empty matrix except for the initial walker."""
    matrix = np.zeros(shape=(matrix_len, matrix_len))
    i, j = initial_walker
    matrix[i][j] = True
    return matrix


def newWalker(matrix, start_i, prob_stick=1):
    """Walk until a walker reaches the cluster.

    Initial walker position is a random cell in the given row start_i. If a
    walker reaches the top or bottom boundary a new walker is started. The
    horizontal boundary is periodic.

    """
    matrix_len = len(matrix)

    def outOfBounds(i):
        """Is the given walker out of bounds?"""
        return i < 0 or i >= matrix_len

    def atCluster(i, j):
        """Has the given walker reached a cluster?"""
        for di, dj in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            if (not outOfBounds(i + di)
                and matrix[i + di][(j + dj) % matrix_len]):
                return True
        return False

    def newPosition(i, j):
        """Walk one cell, left, right, up or down.

        The new cell will not be out of bounds, will not be already be a sink
        and will not be above the starting position start_i.

        """
        deltas = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        di, dj = deltas[np.random.choice(len(deltas))]
        new_i, new_j = i + di, (j + dj) % matrix_len
        if (outOfBounds(new_i) or matrix[new_i][new_j] or new_i < start_i):
            return newPosition(i, j)
        return (new_i, new_j)

    # Create a new walker until one reaches the cluster.
    while True:
        i, j = (start_i, np.random.randint(matrix_len))
        p("Walker starting at ({}, {})".format(i, j))
        # Walk until out of bounds or cluster reached.
        while not outOfBounds(i):
            # Join a cluster with a probability.
            if (atCluster(i, j)
                and (prob_stick >= 1 or np.random.uniform() < prob_stick)):
                p("Walker joined cluster at ({}, {})".format(i, j))
                return (i, j)
            i, j = newPosition(i, j)
            p("Walker moved to = ({}, {})".format(i, j))


def walkerSimulation(matrix_len, initial_walker, max_walkers, prob_stick,
                     save=True, load=False, show=False):
    """Return a matrix after a random walker simulation."""
    fname = "len-{}-init-{}-max-{}-ps-{}.csv".format(
        matrix_len, initial_walker, max_walkers, prob_stick)

    matrix = None
    # Load and return previous results.
    if load:
        try:
            with open(fname) as f:
                matrix = np.loadtxt(f)
                print("Loaded matrix from {}".format(fname))
        except FileNotFoundError:
            print("Could not load matrix from {}".format(fname))

    # If matrix not loaded form file, run the simulation.
    if matrix is None:
        matrix = initialWalkerMatrix(matrix_len, initial_walker)
        start_i = initial_walker[0] - 1  # Start row above initial walker.
        print("start _i = {}".format(start_i))
        p("Initial walker matrix = \n{}".format(matrix))
        for t in range(max_walkers):
            i, j = newWalker(matrix, start_i=start_i, prob_stick=prob_stick)
            print("walker {} found cluster, start_i = {}, i = {}".format(
                t, start_i, i))
            matrix[i][j] = True
            # Start random walker row above the growth.
            start_i = min(max(0, i - 1), start_i)
            p("Matrix after time {} =\n{}".format(t, matrix))
            if start_i == 0:
                print("Growth reached the top")
                break

    # Save the result.
    if save:
        with open(fname, "w") as f:
            np.savetxt(f, matrix)
        print("Saved matrix to {}".format(fname))

    # Plot and save/show result.
    plt.imshow(matrix)
    if show:
        plt.show()
    if save:
        fname = "results/{}.png".format(fname[:-4])
        plt.title("{} random walkers, ps = {}".format(
            int(np.sum(matrix) - 1), prob_stick))
        plt.xlabel("Spatial dimension x")
        plt.ylabel("Spatial dimension y")
        plt.savefig(fname)
        print("Saved plot to {}".format(fname))

    return matrix


if __name__ == "__main__":
    matrix_len = 100
    initial_walker = (matrix_len - 1, int(matrix_len / 2))
    max_walkers = 1000
    probs_stick = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for prob_stick in probs_stick:
        matrix = walkerSimulation(
            matrix_len,
            initial_walker,
            max_walkers,
            prob_stick=prob_stick,
            save=True,
            load=True,
            show=False
        )
