import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation


def diffusion_func(grid, pt, i, j, D, N, delta_t, dx):
    """The concentration based on previous time pt and position (i, j)."""
    # Top boundary condition.
    if (j == N-1):
        return 1
    # Bottom boundary condition.
    if (j == 0):
        return 0
    return (
        grid[pt][i][j] + (delta_t * D / dx ** 2) * (
            # Wrap around on the x-axis.
            grid[pt][0 if i == N-1 else i+1][j]
            + grid[pt][N-1 if i == 0 else i-1][j]
            + grid[pt][i][j+1]
            + grid[pt][i][j-1]
            - 4 * grid[pt][i][j]
        )
    )


def getSimulation(D, N, iterations):
    """Return a matrix indexed first by time, then i and j."""
    dx = 1 / N
    dy = 1 / N
    delta_t = 1 / iterations
    grid = np.zeros(shape=(iterations, N, N))
    # Run the simulation for every time step.
    for t in range(1, iterations):
        for i in range(N):
            for j in range(N):
                grid[t][i][j] = diffusion_func(grid, t-1, i, j, D, N, delta_t, dx)
    return grid


def makeAnimation(grid, show=False, save_times=[0.001, 0.01, 0.1, 1.0]):
    """Generate and show an animation of diffusion."""
    im = plt.imshow(np.flipud(grid[0].T))
    m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array(grid[-1])
    plt.colorbar(
        m,
        label="Concentration",
        boundaries=np.arange(0, 1.01, .01),
        ticks=[0, 1]
    )
    plt.xlabel("x")
    plt.ylabel("y")
    save_indices = [
        min(len(grid) - 1, int(save_time * len(grid) ))
        for save_time in save_times]
    print("Saving animation at index {}".format(save_indices))
    def animate(t):
        nonlocal im
        im.remove()
        im = plt.imshow(np.flipud(grid[t].T), cmap=cm.coolwarm)
        plt.title("t = {0:.3f}".format(t / len(grid)))
        if t in save_indices:
            plt.savefig("animation-t-{}.png".format(t))
    ani = FuncAnimation(plt.gcf(), animate, frames=len(grid), interval=1)
    if show:
        plt.show()


def plotAtTimes(grid, N, times=[0.001, 0.01, 0.1, 1.0], show=False, save=False):
    # A plot of concentration at each y-value, for a few different times.
    js = list(range(len(grid[0])))
    for t in times:
        # Convert time fraction to index.
        timestep = int(t * (len(grid) - 1))
        # Concentration at each vertical for j=0.
        cs = [grid[timestep][0][j] for j in js]
        plt.plot(js, cs, label="t = {0:.3f}".format(t))
    plt.xlabel("Concentration")
    plt.ylabel("Height")
    plt.legend()
    if save:
        plt.savefig("concentration-plot.png", bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    grid = getSimulation(D=1, N=20, iterations=1000)
    makeAnimation(grid, show=True)
    plotAtTimes(grid, N=10, show=False, save=True)
# print(xs)
# print(int(timesteps * 0.5))
# print(grid[int(timesteps * 0.5)])
# for i in range(10):
#     print(grid[i])
