import heapq
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.animation import FuncAnimation, writers


def makeMatrixM(max_k, max_l, boundary_func):
    """Construct a matrix M of given size and shape."""
    M = np.zeros((max_k * max_l, max_k * max_l))
    for k in range(max_k):
        for l in range(max_l):
            if not boundary_func(k, l, max_k, max_l):
                M[k * max_l + l, k * max_l + l] = -4
                M[k * max_l + l, (k + 1) * max_l + l] = 1
                M[k * max_l + l, (k - 1) * max_l + l] = 1
                M[k * max_l + l, k * max_l + (l + 1)] = 1
                M[k * max_l + l, k * max_l + (l - 1)] = 1
    return M


def boundary(k, l, max_k, max_l):
    """Is the given point outside the boundary?"""
    return (k == 0 or k == max_k - 1
           or l == 0 or l == max_l - 1)


def circleBoundary(k, l, max_k, max_l):
    """Is the given point outside the circle boundary?"""
    if max_k % 2 == 0 or max_l % 2 == 0:
        raise ValueError("Circle does not have odd length")
    if max_k != max_l:
        raise ValueError("Circle height not equal to width")
    if boundary(k, l, max_k, max_l):
        return True
    C = max_k // 2
    dist = np.sqrt(abs(k - C) ** 2 + abs(l - C) ** 2)
    r = (max_k - 2) / 2
    return dist > r


def plotMatrixM(M, max_k, max_l, fstart="", show=True, save=True):
    """Plot the given matrix M using plt.imshow."""
    plt.imshow(M.T)
    plt.title("Matrix M for a {} x {} system".format(
        max_k, max_l))
    plt.ylabel("Index k")
    plt.xlabel("Index l")
    plt.colorbar()
    if save:
        plt.savefig("results/{0}matrix-m-{1}x{1}.png".format(
            fstart, max_k, max_l))
    if show:
        plt.show()


def get_smallest_eigenvalues(eigenvalues, n=10):
    """Return a list of (eigenvalue, index) of the n smallest eigenvalues."""
    return heapq.nsmallest(
        n,
        list(enumerate(eigenvalues)),
        key=lambda x: abs(x[1])
    )


def plot_spectrum_of_eigen_frequencies(Nis, Njs, boundary_, shape,
                                       load=True, save=True, show=True,
                                       y_lim_top=None, fappend=""):
    """3D: Plot eigenfrequencies while varying discretization step."""
    plt.close()
    print("Nis = {}".format(Nis))
    print("Njs = {}".format(Njs))
    L = 1
    eigenFrequencies = []
    for i in range(len(Nis)):
        Ni = Nis[i]
        Nj = Njs[i]
        print("Ni = {} Nj".format(Ni, Nj))
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
            M = makeMatrixM(Ni + 2, Nj + 2, boundary_)
            dx = L / Ni
            answer = np.linalg.eig(M * 1/dx**2)
            eigenvalues = np.sort([x for x in answer[0] if x != 0])
        # plotMatrixM(M, Ni=Ni, Nj=Nj, show=True, save=False)
        eigenFrequencies.append(eigenvalues)
        if save:
            with open(fname, "wb") as f:
                pickle.dump(eigenvalues, f)
    lam = [np.sqrt(k * -1) for k in eigenFrequencies]
    plt.boxplot(lam, labels=[str(Nis[i]) for i in range(len(Nis))])
    plt.title(("Eigenfrequencies varying discretization step" +
               "\nfor a {} system".format(shape)))
    plt.xlabel("System length (Ni)")
    plt.ylabel("Eigenfrequencies")
    plt.gca().set_ylim(bottom=0)
    if y_lim_top is not None:
        plt.gca().set_ylim(top=y_lim_top)
    if save:
        plt.savefig("results/3d-eigen-frequencies-{}-{}.png".format(
            shape, fappend))
    if show:
        plt.show()


def plot_eigenvectors_for_shapes(Ni_, Nj_, save=True, show=True):
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
        plotMatrixM(M, Ni, Nj, save=False, show=show)

        eigenvalues_and_eigenvectors = np.linalg.eig(M * 1/dx**2)
        eigenvalues = [x for x in eigenvalues_and_eigenvectors[0] if x != 0]
        eigenvectors = eigenvalues_and_eigenvectors[1].T
        smallest_eigenvalues = get_smallest_eigenvalues(eigenvalues, n=20)

        # For each of n smallest eigenvalues.
        for i, k in smallest_eigenvalues:
            eigenvalue = np.sqrt(k * -1)

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
                json.dump(eigenvector.real.tolist(), f)

            # Plot subplot.
            plt.imshow(eigenmatrix.real)
            plt.ylabel("Index k")
            plt.xlabel("Index l")
            plt.title(("Reshaped eigenvector for λ = {:.2f}\n " +
                        "of a {} system on a {} x {} grid").format(
                            eigenvalue.real, shape, Ni + 2, Nj + 2))

            # Save/show figure.
            if save:
                plt.savefig(("results/3b-eigenvector-" +
                            "shape-{}-eigenvector{:.2f}").format(
                                shape, eigenvalue).replace(".", "-") + ".png")
            if show:
                plt.show()


def animation_of_membrane(v0, K, shape, Ni, Nj,
                          A=1, B=1, c=1, dt=0.01, max_t=6.4,
                          show=True, save=True):
    """Animation for given initial vector v0 and constants."""
    print("shape = {} Ni = {} Nj = {}".format(shape, Ni, Nj))

    lam = np.sqrt(K * -1)
    v = v0
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
    im = plt.imshow(np.array(v).reshape(Ni + 2, Nj + 2), norm=norm)

    # Title and colorbar.
    plt.title("Animation for λ = {:.2f}\nfor a {} system".format(
        lam, shape))
    plt.colorbar(norm=norm)

    # Plot later frame.
    def animate(t):
        nonlocal im
        im.remove()
        new_v = np.array(v) * T(t)
        im = plt.imshow(new_v.reshape(Ni + 2, Nj + 2), norm=norm)

    indices = np.arange(0, max_t, dt)
    ani = FuncAnimation(plt.gcf(), animate, frames=indices, interval=1)

    if show:
        plt.show()
    if save:
        Writer = writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        fname = "results/animation-λ={}-shape-{}.mp4".format(lam, shape)
        ani.save(fname, writer=writer)
        print("Saved {}".format(fname))


if __name__ == "__main__":
    # Question A.
    Ni, Nj = 4, 4
    M = makeMatrixM(Ni + 2, Nj + 2, boundary)
    plotMatrixM(M, Ni, Nj, fstart="3a-", show=True)

    # Question B.
    plot_eigenvectors_for_shapes(Ni_=49, Nj_=49, show=True)

    # Question D.
    for shape, Nis, boundary_ in [
            ("square", np.arange(9, 60, 10), boundary),
            ("rectangular", np.arange(9, 60, 10), boundary),
            ("circular", np.arange(9, 60, 10), circleBoundary)
    ]:
        plot_spectrum_of_eigen_frequencies(
            Nis=Nis, Njs=(Nis * 2) if shape == "rectangular" else Nis,
            boundary_=boundary_, shape=shape, load=True, save=True, show=True
        )
        plot_spectrum_of_eigen_frequencies(
            Nis=Nis, Njs=(Nis * 2) if shape == "rectangular" else Nis,
            boundary_=boundary_, shape=shape, load=True, save=True, show=True,
            y_lim_top=20, fappend="ylim"
        )

    # Question E.
    from e_params import params
    for shape, k, Ni, Nj, eigen_vector in params:
        animation_of_membrane(eigen_vector, k, shape, Ni, Nj, show=True, save=True)
