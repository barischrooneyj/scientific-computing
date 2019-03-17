import heapq
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np


def makeMatrixM(Ni, Nj, boundary_func):
    M = np.zeros((Ni * Nj, Ni * Nj))

    for i in range(Ni):
        for j in range(Nj):
            # print("i, j = {}".format(i, j))
            if not boundary_func(i, j, Ni, Nj):

                # print(i , j , "Special")
                M[i * Ni + j, i * Ni + j] = -4
                M[i * Ni + j, (i + 1) * Ni + j] = 1
                M[i * Ni + j, (i - 1) * Ni + j] = 1
                M[i * Ni + j, i * Ni + (j + 1)] = 1
                M[i * Ni + j, i * Ni + (j - 1)] = 1
    return M


def boundary(i, j, Ni, Nj):
    if i == 0 or i == Ni - 1:
        return True

    elif j == 0 or j == Nj - 1:
        return True

    else:
        return False


def circleBoundary(i, j, Ni, Nj):
    if Ni % 2 == 0:
        raise ValueError("Circle does not have odd length")
    if boundary(i, j, Ni, Nj):
        return True
    C = Ni // 2
    dist = np.sqrt(abs(i - C) ** 2 + abs(j - C) ** 2)
    r = (Ni-2) / 2
    return dist > r


def smallest_eigenvalues(eigenvalues, n=10):
    return heapq.nsmallest(
        n,
        list(enumerate(eigenvalues)),
        key=lambda x: abs(x[1])
    )


def plotMatrixM(M):
    plt.imshow(M.T)
    plt.title("Matrix M from Equation X")
    plt.ylabel("Index i")
    plt.xlabel("Index j")
    plt.savefig("results/matrix-m-{0}x{0}.png".format(Ni, Nj))
    plt.show()


def plotSpectrumOfEigenFrequencies(
        Nis=np.arange(9, 80, 10), load=True, save=True, show=True):
    """Plot eigenfrequencies while varying discretization step."""
    print("Nis = {}".format(Nis))
    L = 1
    eigenFrequencies = []
    for Ni in Nis:
        print("Ni = {}".format(Ni))
        fname = "data/eigenfrequencies-{}.pickle".format(Ni)
        loaded = False
        if load:
            try:
                with open(fname, "rb") as f:
                    eigenvalues = pickle.load(f)
                loaded = True
            except:
                print("Could not load {}".format(fname))
        if not loaded:
            M = makeMatrixM(Ni + 2, Ni + 2, boundary)
            dx = L / Ni
            answer = np.linalg.eig(M * 1/dx**2)
            eigenvalues = np.sort([x for x in answer[0] if x != 0])
        eigenFrequencies.append(eigenvalues)
        if save:
            with open(fname, "wb") as f:
                pickle.dump(eigenvalues, f)
    plt.boxplot(eigenFrequencies, labels=[str(x) for x in Nis])
    # plt.yscale("log")
    plt.title(
        "Eigenfrequencies varying discretization step")
    if save:
        plt.savefig("results/eigen-frequencies-{}.png".format(Nis).strip())
    plt.show()


if __name__ == "__main__":
    # Question D.
    plotSpectrumOfEigenFrequencies(load=True, save=True)

    # L = 1
    # Ni = 21
    # Nj = 21
    # dx = L / Ni
    # dy = L / Nj

    # M = makeMatrixM(Ni + 2, Nj + 2, boundary)
    # plotMatrixM(M)
    # M = makeMatrixM(Ni + 2, 2 * Nj + 2, boundary)
    # plotMatrixM(M)
    # M = makeMatrixM(Ni + 2, Nj + 2, circleBoundary)
    # plotMatrixM(M)

    # v = np.zeros((Nj * Ni, 1))

    # answer = np.linalg.eig(M * 1/dx**2)
    # eigenvalues = [x for x in answer[0] if x != 0]
    # print(eigenvalues)
    # eigenvectors = answer[1].T
    # smallest = smallest_eigenvalues(eigenvalues, n=10)

    # for i, eigenvalue in smallest:
    #     eigenvector = eigenvectors[i]
    #     eigenmatrix = eigenvector.reshape(Ni+2, Nj+2)
    #     plt.imshow(eigenmatrix.real)
    #     plt.show()
        # plt.plot(list(range(len(eigenvector))), eigenvector)
        # plt.show()

    # print(answer[0])
    # print()
    # print(answer[1])

    #for i in range(len(answer[0])):
    #    if abs(answer[0][i]) <0.01:
    #        print(answer[0][i], answer[1][i], np.dot(M, answer[1][i]))
