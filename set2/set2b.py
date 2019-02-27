import numpy as np

# Top row i = 0.
# Left column j = 0.

PRINT = True


def p(s):
    """Print if PRINT is true."""
    if PRINT:
        print(s)


def initialWalkerMatrix(matrix_len, initial_walker):
    """An empty matrix except for the initial walker."""
    matrix = np.zeros(shape=(matrix_len, matrix_len))
    i, j = initial_walker
    matrix[i][j] = True
    return matrix


def newWalker(matrix, prob_stick=1):
    """Walk until a walker reaches the cluster.

    Initial walker position is a random cell in the top row. If a walker
    reaches the top or bottom boundary a new walker is started. The horizontal
    boundary is periodic.
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
        """Walk one cell, left, right, up or down."""
        deltas = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        di, dj = deltas[np.random.choice(len(deltas))]
        return (i + di, (j + dj) % matrix_len)

    # Create a new walker until one reaches the cluster.
    while True:
        i, j = (0, np.random.randint(matrix_len))
        p("Walker starting at ({}, {})".format(i, j))
        # Walk until out of bounds or cluster reached.
        while not outOfBounds(i):
            if (atCluster(i, j)
                and (prob_stick >= 1 or np.random.uniform() < prob_stick)):
                p("Walker joined cluster at ({}, {})".format(i, j))
                return (i, j)
            i, j = newPosition(i, j)
            p("Walker moved to = ({}, {})".format(i, j))


def walkerSimulation(matrix_len, initial_walker, max_walkers, prob_stick,
                     save=True, load=False):
    """Return a matrix after a random walker simulation."""
    fname = "len-{}-init-{}-max-{}-ps-{}.csv".format(
        matrix_len, initial_walker, max_walkers, prob_stick)

    # Load and return previous results if possible.
    if load:
        try:
            with open(fname) as f:
                print("Loaded matrix from {}".format(fname))
                return np.loadtxt(f)
        except FileNotFoundError:
            print("Could not load matrix from {}".format(fname))

    matrix = initialWalkerMatrix(matrix_len, initial_walker)
    p("Initial walker matrix = \n{}".format(matrix))
    for t in range(max_walkers):
        p("Looking for walker at time {}".format(t))
        i, j = newWalker(matrix, prob_stick)
        matrix[i][j] = True
        p("Matrix after time {} =\n{}".format(t, matrix))

    # Save the result
    if save:
        with open(fname, "w") as f:
            np.savetxt(f, matrix)
        print("Saved matrix to {}".format(fname))

    return matrix


if __name__ == "__main__":
    matrix_len = 100
    initial_walker = (-1, 49)
    max_walkers = 1000
    probs_stick = [0.1, 0.3, 0.5, 0.7, 0.9, 1]

    for prob_stick in probs_stick:
        matrix = walkerSimulation(
            matrix_len,
            initial_walker,
            max_walkers,
            prob_stick=prob_stick,
            save=True,
            load=False
        )
