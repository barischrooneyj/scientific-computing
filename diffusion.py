import itertools
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation, writers


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


def getSimulation(D, N, delta_t, final_time=None, terminate_when=None):
    """Return a matrix indexed first by time, then i and j.

    Either run for final_time/delta_t iterations, or if terminate_when is given
    then run until the condition is met.
    """
    dx = 1 / N
    dy = 1 / N

    # Fixed amount of iterations if final_time given, else we will use a
    # termination condition.
    iterations = np.inf
    if final_time:
        iterations = final_time / delta_t
        if int(iterations) != iterations:
            raise ValueError("final_time not divisible by delta_t")
        iterations = int(iterations)
        print("delta_t = {} final_time = {} iterations = {}".format(
            delta_t, final_time, iterations))
    else:
        print("delta_t = {}".format(delta_t, final_time, iterations))

    grid = np.zeros(shape=(128, N, N))
    # Run the simulation for every time step.
    for t in itertools.count():
        # Stop when final index reached or termination condition.
        if (final_time and t == iterations - 1
            or terminate_when and t > 2 and terminate_when(grid[t - 1], grid[t - 2])):
            break
        # Resize array if size reached.
        if grid.shape[0] - 1 == t:
            grid.resize(grid.shape[0] * 2, N, N)
        # Calculate concentration for every cell.
        for i in range(N):
            for j in range(N):
                grid[t][i][j] = diffusion_func(
                    grid, t-1, i, j, D, N, delta_t, dx)
    return grid[:t + 1]


def makeAnimation(grid, show=False, save=False,
                  save_times=[0.001, 0.01, 0.1, 1.0]):
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
    print("Saving animation at times {}".format(save_times))
    def animate(t):
        nonlocal im
        im.remove()
        im = plt.imshow(np.flipud(grid[t].T), cmap=cm.coolwarm)
        plt.title("t = {0:.3f}".format(save_times[save_indices.index(t)]))
        if t in save_indices:
            plt.savefig("animation-t-{}.png".format(
                save_times[save_indices.index(t)]))
    print(len(grid))
    ani = FuncAnimation(plt.gcf(), animate, frames=len(grid), interval=1)
    if show:
        plt.show()
    if save:
        Writer = writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("diffusion-animation.mp4", writer=writer)
        print("Saved diffusion-animation.mp4")


def plotAtTimes(grid, N, times=[0.001, 0.01, 0.1, 1.0], show=False, save=False):
    # A plot of concentration at each y-value, for a few different times.
    plt.close()
    js = list(range(len(grid[0])))
    print("Plotting concentration at times {}".format(times))
    for t in times:
        # Convert time fraction to index.
        timestep = int(t * (len(grid) - 1))
        # Round up 0.
        if timestep < 2:
            timestep = 2
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
    grid = getSimulation(D=1, N=20, delta_t=0.001, final_time=1)
    # two_epsilon = lambda a, b: (a - b < 2 * np.finfo(np.float32).eps).all()
    # grid = getSimulation(D=1, N=20, delta_t=0.001, terminate_when=two_epsilon)
    makeAnimation(grid, show=True, save=False)
    # plotAtTimes(grid, N=10, show=False, save=True)
