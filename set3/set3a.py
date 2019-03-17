import heapq
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np


def makeMatrixM(Ni, Nj, boundary_func):
    M = np.zeros((Ni * Nj, Ni * Nj))
    for i in range(Ni):
        for j in range(Nj):
            if not boundary_func(i, j, Ni, Nj):
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
    if Ni % 2 == 0 or Nj % 2 == 0:
        raise ValueError("Circle does not have odd length")
    if Ni != Nj:
        raise ValueError("Circle height not equal to width")
    if boundary(i, j, Ni, Nj):
        return True
    C = Ni // 2
    dist = np.sqrt(abs(i - C) ** 2 + abs(j - C) ** 2)
    r = (Ni-2) / 2
    return dist > r


def plotMatrixM(M, Ni, Nj, fstart="", show=True, save=True):
    """Plot the given matrix M using plt.imshow."""
    plt.imshow(M.T)
    plt.title("Matrix M from Equation X")
    plt.ylabel("Index i")
    plt.xlabel("Index j")
    if save:
        plt.savefig("results/{0}matrix-m-{1}x{1}.png".format(fstart, Ni, Nj))
    if show:
        plt.show()


def get_smallest_eigenvalues(eigenvalues, n=10):
    """Return a list of (eigenvalue, index) of the n smallest eigenvalues."""
    return heapq.nsmallest(
        n,
        list(enumerate(eigenvalues)),
        key=lambda x: abs(x[1])
    )


def plot_spectrum_of_eigen_frequencies(
        Nis=np.arange(9, 90, 10), load=True, save=True, show=True):
    """3D: Plot eigenfrequencies while varying discretization step."""
    print("Nis = {}".format(Nis))
    L = 1
    eigenFrequencies = []
    for Ni in Nis:
        print("Ni = {}".format(Ni))
        fname = "data/3d-eigenfrequencies-{}.pickle".format(Ni)
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
    plt.title("Eigenfrequencies varying discretization step")
    plt.xlabel("Ni (Maximum space index i)")
    plt.ylabel("Eigenfrequencies")
    locs, _ = plt.yticks()
    labels = ["0" if l == 0 else "{:.0E}".format(l) for l in locs]
    plt.yticks(locs, labels)
    if save:
        plt.savefig("results/3d-eigen-frequencies.png")
    if show:
        plt.show()


def plot_eigenvectors_for_shapes(Ni_=29, Nj_=29, save=True, show=True):
    """3B: Plot eigenvectors for the smallest eigenvalues for each shape."""
    if Ni_ % 2 == 0:
        raise ValueError("Ni should be an odd number")
    if Nj_ % 2 == 0:
        raise ValueError("Nj should be an odd number")

    L = 1
    dx = L / Ni_
    dy = L / Nj_

    for shape, Ni, Nj, boundary_f in [
        ("square",      Ni_, Nj_,   boundary),
        ("rectangular", Ni_, 2*Nj_, boundary),
        ("circular",    Ni_, Nj_,   circleBoundary)
    ]:
        M = makeMatrixM(Ni + 2, Nj + 2, boundary_f)
        plotMatrixM(M, Ni, Nj, save=False, show=show)

        eigenvalues_and_eigenvectors = np.linalg.eig(M * 1/dx**2)
        eigenvalues = [x for x in eigenvalues_and_eigenvectors[0] if x != 0]
        eigenvectors = eigenvalues_and_eigenvectors[1].T
        smallest_eigenvalues = get_smallest_eigenvalues(eigenvalues, n=10)

        for i, eigenvalue in smallest_eigenvalues:
            eigenvector = eigenvectors[i]
            eigenmatrix = eigenvector.reshape(Ni + 2, Nj + 2)
            print("Eigenvalue {} for shape {}".format(eigenvalue, shape))
            plt.imshow(eigenmatrix.real)
            plt.ylabel("Row index i")
            plt.xlabel("Column index j")
            plt.title(
                "Eigenvector (reshaped) for eigenvalue"
                + "\n{:.2f} for a {} x {} {} system".format(
                    eigenvalue, Ni, Nj, shape)
            )
            if save:
                plt.savefig("results/3b-eigenvector-shape-{}-eigenvector{:.2f}".format(
                    shape, eigenvalue).replace(".", "-") + ".png")
            if show:
                plt.show()


if __name__ == "__main__":
    # Question A.
    Ni, Nj = 4, 4
    M = makeMatrixM(Ni + 2, Nj + 2, boundary)
    plotMatrixM(M, Ni, Nj, fstart="3a-", show=True)
    # Question B.
    plot_eigenvectors_for_shapes(show=True)
    # Question D.
    plot_spectrum_of_eigen_frequencies(load=True, save=True, show=True)





    # plt.plot(list(range(len(eigenvector))), eigenvector)
    # plt.show()

    # print(answer[0])
    # print()
    # print(answer[1])

    #for i in range(len(answer[0])):
    #    if abs(answer[0][i]) <0.01:
    #        print(answer[0][i], answer[1][i], np.dot(M, answer[1][i]))
