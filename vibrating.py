import math
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation

def Phifunction(x):
    return math.sin(2 * math.pi * x)

L = 1
N = 500
endTime = 800
c = 1
deltaT = 0.001
frameSkip = 5

# Setup a string matrix, a wave amplitude at each position at each time step.
# Initially we have only two time steps with values determined from t=0.
stringMatrix = []
startString = [0] * (N+1)

for i in range(N+1):
    if (i == 0 or i == N):
        pass
    else:
        startString[i] = Phifunction(L/N*i)

stringMatrix.append(startString)
stringMatrix.append(startString)

# For the remaining time steps calculate wave amplitude at each position.
for j in range(endTime):
    tempString = [0] * (N+1)
    for i in range(N + 1):
        if(i == 0 or i == N):
            tempString[i] = 0
        else:
            tempString[i] = (
                (deltaT * c / (L/N) ) ** 2
                * (stringMatrix[-1][i+1] + stringMatrix[-1][i-1]
                   - 2 * stringMatrix[-1][i])
                - stringMatrix[-2][i] + 2 * stringMatrix[-1][i])
    stringMatrix.append(tempString)

# Plot a repeating animation of the wave over time.
x = range(N + 1)
line, = plt.plot(x, stringMatrix[0])
def animate(i):
    global line
    line.remove()
    line, = plt.plot(x, stringMatrix[i])
ani = FuncAnimation(
    plt.gcf(), animate, frames=range(0, len(stringMatrix), frameSkip),
    interval=1)
plt.show()
