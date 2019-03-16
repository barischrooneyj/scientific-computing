import heapq

import matplotlib.pyplot as plt
import numpy as np

def makeMatrixM(Ni, Nj):
    M = np.zeros((Ni * Nj, Ni * Nj))

    for i in range(Ni):
        for j in range(Nj):
            if not boundary(i, j, Ni, Nj):

                print(i , j , "Special")
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


def smallest_eigenvalues(eigenvalues, n=10):
    return heapq.nsmallest(
        n,
        list(enumerate(eigenvalues)),
        key=lambda x: abs(x[1])
    )


if __name__ == "__main__":
    L = 1
    Ni = 6
    Nj = 6
    dx = L / Ni
    dy = L / Nj

    M = makeMatrixM(Ni + 2, Nj + 2)
    v = np.zeros((Nj * Ni, 1))
    plt.imshow(M.T)
    plt.title("Matrix M from Equation X")
    plt.ylabel("Index i")
    plt.xlabel("Index j")
    plt.savefig("results/matrix-m-{0}x{0}.png".format(Ni, Nj))
    plt.show()

    answer = np.linalg.eig(M * 1/dx**2)
    eigenvalues = [x for x in answer[0] if x != 0]
    print(eigenvalues)
    eigenvectors = answer[1].T
    smallest = smallest_eigenvalues(eigenvalues, n=10)

    for i, eigenvalue in smallest:
        eigenvector = eigenvectors[i]
        eigenmatrix = eigenvector.reshape(Ni+2, Nj+2)
        plt.imshow(eigenmatrix.real)
        plt.show()
        # plt.plot(list(range(len(eigenvector))), eigenvector)
        # plt.show()

    # print(answer[0])
    # print()
    # print(answer[1])

    #for i in range(len(answer[0])):
    #    if abs(answer[0][i]) <0.01:
    #        print(answer[0][i], answer[1][i], np.dot(M, answer[1][i]))
