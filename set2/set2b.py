import numpy as np

# Top row i = 0.
# Left column j = 0.

DEBUG = True


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


def newWalker(matrix):
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
        p("New walker at position ({}, {})".format(i, j))
        # Walk until out of bounds or cluster reached.
        while not outOfBounds(i):
            if atCluster(i, j):
                p("New walker at ({}, {})".format(i, j))
                return (i, j)
            i, j = newPosition(i, j)
            p("New position = ({}, {})".format(i, j))


def walkerSimulation(matrix_len, initial_walker, max_walkers):
    """Return a matrix after a random walker simulation."""
    p("Initial walker = {}".format(initial_walker))
    matrix = initialWalkerMatrix(matrix_len, initial_walker)
    p("Initial walker matrix = \n{}".format(matrix))
    for t in range(max_walkers):
        p("Looking for walker at time {}".format(t))
        i, j = newWalker(matrix)
        matrix[i][j] = True
        p("Matrix after time {} =\n{}".format(t, matrix))
    return matrix


if __name__ == "__main__":
    walkerSimulation(20, (-1, 9), 100)
