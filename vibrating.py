import math
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.animation import FuncAnimation, writers


def getStringMatrix(L=1, N=500, endTime=800, c=1, deltaT=0.001,
                    phi_func=lambda x: math.sin(2 * math.pi * x)):
    """A matrix, indexed first by time step then by x coordinate."""

    stringMatrix = []
    startString = [0] * (N+1)

    for i in range(N+1):
        if (i == 0 or i == N):
            pass
        else:
            startString[i] = phi_func(L/N*i)

    # Initially we have only two time steps with values determined from t=0.
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

    return stringMatrix



def animate(stringMatrix, N=500, frameSkip=5):
    """Plot a repeating animation of the wave over time.
    Args:
        N: the number of points on the x-axis.
    """

    colors = [cm.jet(x) for x in np.linspace(0, 1, len(stringMatrix))]
    x = range(N + 1)
    line, = plt.plot(x, stringMatrix[0])
    def animate(i):
        nonlocal line
        line.remove()
        line, = plt.plot(x, stringMatrix[i], color=colors[i])
    ani = FuncAnimation(
        plt.gcf(), animate, frames=range(0, len(stringMatrix), frameSkip),
        interval=1)
    return ani


if __name__ == "__main__":
    titles_and_phi_funcs = [
        ("1-1-B-i", "sin(2πx)", lambda x: math.sin(2 * math.pi * x)),
        ("1-1-B-ii", "sin(5πx)", lambda x: math.sin(5 * math.pi * x)),
        ("1-1-B-iii", "sin(5πx) if 1/5 < x < 2/5 else 0",
         lambda x: math.sin(5 * math.pi * x) if (0.2 < x and 0.4 > x) else 0)
    ]

    # Generate a plot for each of the three setups.
    time_fractions = np.linspace(0, 1, 30)

    # For each Phi function.
    for (filename, title, phi_func) in titles_and_phi_funcs:
        stringMatrix = getStringMatrix(phi_func=phi_func)
        colors = [cm.jet(x) for x in np.linspace(0.2, 0.8, len(stringMatrix))]
        # Plot for each time step.
        for time_fraction in time_fractions:
            time_step = int((len(stringMatrix) - 1) * time_fraction)
            plt.plot(
                range(len(stringMatrix[0])),
                stringMatrix[time_step],
                label=(int(time_fraction)
                       if time_step == 0 or time_step == (len(stringMatrix) - 1)
                       else None),
                color=colors[time_step]
            )
        plt.legend()
        plt.title(title)
        plt.savefig(filename + ".png", bbox_inches="tight")
        print("Saved {}.png".format(filename))
        plt.close()

    Writer = writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # For each Phi function generate an animation.
    for (filename, title, phi_func) in titles_and_phi_funcs:
        ani = animate(getStringMatrix(phi_func=phi_func))
        ani.save(filename + ".mp4", writer=writer)
        print("Saved {}.mp4".format(filename))
