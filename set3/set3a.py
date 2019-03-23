import heapq
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.animation import FuncAnimation, writers

def makeMatrixM(Ni, Nj, boundary_func):
    M = np.zeros((Ni * Nj, Ni * Nj))
    for i in range(Ni):
        for j in range(Nj):
            if not boundary_func(i, j, Ni, Nj):
                M[i * Nj + j, i * Nj + j] = -4
                M[i * Nj + j, (i + 1) * Nj + j] = 1
                M[i * Nj + j, (i - 1) * Nj + j] = 1
                M[i * Nj + j, i * Nj + (j + 1)] = 1
                M[i * Nj + j, i * Nj + (j - 1)] = 1
    return M


def boundary(i, j, Ni, Nj):
    return (i == 0 or i == Ni - 1
           or j == 0 or j == Nj - 1)


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
    plt.title("Matrix M from Equation 9 for a {} x {} system".format(Ni, Nj))
    plt.ylabel("Index k of matrix M")
    plt.xlabel("Index l of matrix M")
    plt.colorbar()
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


def plot_eigenvectors_for_shapes(Ni_=29, Nj_=29, save=True, show=True,
                                 plots_per_subplot=4):
    """3B: Plot eigenvectors for the smallest eigenvalues for each shape."""
    if Ni_ % 2 == 0:
        raise ValueError("Ni should be an odd number")
    if Nj_ % 2 == 0:
        raise ValueError("Nj should be an odd number")

    L = 1
    dx = L / Ni_
    dy = L / Nj_

    # For each system shape.
    for shape, Ni, Nj, boundary_f in [
        ("square",      Ni_, Nj_,   boundary),
        ("rectangular", Ni_, 2*Nj_, boundary),
        ("circular",    Ni_, Nj_,   circleBoundary)
    ]:
        M = makeMatrixM(Ni + 2, Nj + 2, boundary_f)
        plt.close()
        plotMatrixM(M, Ni, Nj, save=False, show=show)

        eigenvalues_and_eigenvectors = np.linalg.eig(M * 1/dx**2)
        eigenvalues = [x for x in eigenvalues_and_eigenvectors[0] if x != 0]
        eigenvectors = eigenvalues_and_eigenvectors[1].T
        smallest_eigenvalues = get_smallest_eigenvalues(eigenvalues, n=20)

        eig_names = set()
        def notyet(i, eigenvalue):
            """Eigenvalue not yet seen with same leading integer part?"""
            eig_name = str(eigenvalue).split(".", 1)[0]
            if eig_name in eig_names:
                return False
            eig_names.add(eig_name)
            return True

        # Remove duplicate eigenvalues.
        smallest_eigenvalues = [x for x in smallest_eigenvalues if notyet(*x)]

        plt.close()
        # For each of n smallest eigenvalues.
        for plt_num, (i, eigenvalue) in enumerate(smallest_eigenvalues):

            # Create a new figure for the subplots.
            if plt_num % plots_per_subplot == 0:
                fig, axes = plt.subplots(plots_per_subplot, 1)

            # Reshape eigenvector.
            eigenvector = eigenvectors[i]
            eigenmatrix = eigenvector.reshape(Ni + 2, Nj + 2)

            # Save eigenvector to file.
            fname = "eigenvectors/shape-{}-{}-x-{}-eigenvalue-{}.json".format(
                shape, Ni, Nj, eigenvalue)
            print(("Eigenvector for eigenvalue {}\nfor shape {} " +
                   "system size {} x {} saved to:\n\t{}").format(
                       eigenvalue, shape, Ni, Nj, fname))
            with open(fname, "w") as f:
                json.dump(eigenvector.tolist(), f)

            # Plot subplot.
            ax = axes[plt_num % plots_per_subplot]
            ax.imshow(eigenmatrix.real)
            ax.set_ylabel("Index k")
            ax.set_xlabel("Index l")
            ax.set_title("λ = {:.2f}".format(eigenvalue))

            # Save/show figure.
            if plt_num % plots_per_subplot == plots_per_subplot - 1:

                # TODO: Set subplot width.
                # fig.set_figwidth(5)
                # plt.subplots_adjust(wspace=0)

                plt.suptitle(("Reshaped eigenvector for a\n " +
                              "{} system on a {} x {} grid").format(
                                    shape, Ni + 2, Nj + 2))
                plt.tight_layout()
                if save:
                    plt.savefig(("results/3b-eigenvector-" +
                                "shape-{}-eigenvector{:.2f}").format(
                                shape, eigenvalue).replace(".", "-") + ".png")
                if show:
                    plt.show()


def animation_of_membrane(v0, K, A=1, B=1, c=1, dt=0.01, max_t=6.4,
                          show=True, save=True):
    """Animation for given initial vector v0 and constants."""
    lam = np.sqrt(K * -1)
    v = v0
    X = int(np.sqrt(len(v)))
    T = lambda t: A * np.cos(c * lam * t) + B * np.sin(c * lam * t)

    # First pass find maximum v and minimum v for colormap.
    min_v = 1
    max_v = -1
    for t in np.arange(0, max_t, dt):
        new_v = np.array(v) * T(t)
        if np.min(new_v) < min_v: min_v = np.min(new_v)
        if np.max(new_v) > max_v: max_v = np.max(new_v)
    print("min_v = {}, max_v = {}".format(min_v, max_v))

    # Now make an animation.
    norm = colors.Normalize(vmin=min_v, vmax=max_v)
    # Plot first frame.
    im = plt.imshow(np.array(v).reshape(X, X), norm=norm)

    # Plot later frame.
    def animate(t):
        nonlocal im
        im.remove()
        new_v = np.array(v) * T(t)
        im = plt.imshow(new_v.reshape(X, X), norm=norm)

    indices = np.arange(0, max_t, dt)
    ani = FuncAnimation(plt.gcf(), animate, frames=indices, interval=1)

    if show:
        plt.show()
    if save:
        Writer = writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        fname = "animation-K={}.mp4".format(K)
        ani.save(fname, writer=writer)
        print("Saved {}".format(fname))


if __name__ == "__main__":
    # Question A.
    Ni, Nj = 4, 4
    M = makeMatrixM(Ni + 2, Nj + 2, boundary)
    # plotMatrixM(M, Ni, Nj, fstart="3a-", show=True)

    # Question B.
    plot_eigenvectors_for_shapes(Ni_=29, Nj_=29, show=True)

    # Question D.
    plot_spectrum_of_eigen_frequencies(load=True, save=True, show=True)

    # Question E.
    import e_params
    for K, v0 in [
        (e_params.K0, e_params.v0), (e_params.K1, e_params.v1)
    ]:
        animation_of_membrane(v0, K, show=False, save=True)
