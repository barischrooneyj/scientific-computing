import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def makeStartGrid(matrix_len = 100, u_init = 0.5, v_init = 0.25):
    noise_grid = np.random.normal(scale = 0.1, size=(matrix_len, matrix_len))
    
    u_grid = np.ones(shape=(matrix_len, matrix_len)) * u_init 
    v_grid = np.zeros(shape=(matrix_len, matrix_len)) + noise_grid
    
    v_len = 20
    v_i = (matrix_len // 2) - (v_len // 2)
    v_j = v_i
    print(v_i, v_j)
    for di in range(v_len):
        for dj in range(v_len):
            v_grid[v_i + di][v_j + dj] = v_init
    return u_grid, v_grid

@jit (nopython = True)
def update_grid(u_grid, v_grid, Du = 0.16, Dv = 0.08, f = 0.035, k = 0.06, delta_t = 1, delta_x = 1):

    N = len(u_grid)
    
    new_u_grid = np.zeros(shape=(N, N))
    new_v_grid = np.zeros(shape=(N, N))
    
    for i in range(N):
        for j in range(N):
            new_u_grid[i][j] = (u_grid[i][j] +
                    Du * delta_t / (delta_x **2) *
                    (u_grid[0 if i == N - 1 else i+1][j]
                    + u_grid[N-1 if i == 0 else i-1][j]
                    + u_grid[i][0 if j == N - 1 else j+1]
                    + u_grid[i][N-1 if j == 0 else j-1]
                    - 4 * u_grid[i][j])
                    
                    - u_grid[i][j] * v_grid[i][j]**2
                    + f * (1 - u_grid[i][j]))
                    
            new_v_grid[i][j] = (v_grid[i][j] +
                    Dv * delta_t / (delta_x **2) *
                    (v_grid[0 if i == N - 1 else i+1][j]
                    + v_grid[N-1 if i == 0 else i-1][j]
                    + v_grid[i][0 if j == N - 1 else j+1]
                    + v_grid[i][N-1 if j == 0 else j-1]
                    - 4 * v_grid[i][j]) 
                    
                    +u_grid[i][j] * v_grid[i][j]**2
                    - (f + k) * v_grid[i][j])
                    
    return new_u_grid, new_v_grid
             

from matplotlib import cm
import matplotlib.colors as colors

if __name__ == "__main__":
    u_grid, v_grid = makeStartGrid()
    
    print(v_grid)
    for u in range(4):
        for _ in range(200):
            u_grid, v_grid = update_grid(u_grid, v_grid)
            
        plt.imshow(u_grid / (u_grid + v_grid), norm=colors.Normalize(vmin=0, vmax=1))
        plt.show()
    print(v_grid)

