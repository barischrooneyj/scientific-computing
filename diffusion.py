import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation

D = 1
N = 8  # Length of box.
dx = 1. / N
dy = 1. / N
timesteps = 1000
delta_t = 1 / timesteps
# First indexed by time, then i and j.
grid = np.zeros(shape=(timesteps, N, N))


def diffusion_func(pt, i, j):
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

# Run the simulation for every time step.
for t in range(1, timesteps):
    for i in range(N):
        for j in range(N):
            grid[t][i][j] = diffusion_func(t-1, i, j)

# Plot an animation of the diffusion over time.
im = plt.imshow(np.flipud(grid[0].T))
m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
m.set_array(grid[-1])
plt.colorbar(m, boundaries=np.arange(0, 1.1, .1))
def animate(t):
    global im
    im.remove()
    im = plt.imshow(np.flipud(grid[t].T), cmap=cm.coolwarm)
ani = FuncAnimation(plt.gcf(), animate, frames=len(grid))
plt.show()

# A plot of concentration at each y-value, for a few different times.
# ts = [0.001, 0.01, 0.1, 1.0]
# js = np.arange(0, 1, 0.1)
# print(js)
# for t in ts:
#     timestep = int(t * timesteps)
#     if timestep == timesteps: timestep - 1
#     cs = [grid[timestep][int(j * N)][0] for j in js]
#     plt.plot(js, cs)
# plt.show()
# print(xs)

print(int(timesteps * 0.5))
print(grid[int(timesteps * 0.5)])

# for i in range(10):
#     print(grid[i])
