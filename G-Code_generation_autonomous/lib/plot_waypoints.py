# import libraries
import numpy as np
import matplotlib.pyplot as plt


# plot waypoints
def plot_waypoints(waypoints, lxy, theta, L, W):
    theta_rad = theta * np.pi / 180
    wh = lxy / np.sin(theta_rad)
    wv = lxy / np.cos(theta_rad)
    # ----------- plot figure ---------------
    plt.figure(figsize=(8, 8))
    for i in range(len(waypoints) - 1):
        point1 = [waypoints[i][0], waypoints[i + 1][0]]
        point2 = [waypoints[i][1], waypoints[i + 1][1]]
        plt.plot(point1, point2, '-o', lw=2.0, color='blue')

    for i in range(100):
        x_o = np.linspace(0, 34, 100)
        y_o = i * wv - (wv / wh) * x_o
        p = plt.plot(x_o, y_o, '--', color='black', alpha=0.25)
        plt.title(r'$\theta = {}$'.format(theta))

    #     plt.hlines(0, 0, L, lw = 3.0, color='red')
    #     plt.vlines(0, 0, W, lw = 3.0, color='red')
    #     plt.hlines(L, 0, L, lw = 3.0, color='red')
    #     plt.vlines(W, 0, W, lw = 3.0, color='red')

    x = [i for i in range(35)]
    plt.xticks(x, x)
    x = [i for i in range(35)]
    plt.yticks(x, x)

    plt.xlim([-1, L + 1])
    plt.ylim([-1, W + 1])
    plt.savefig('p_{}.png'.format(theta), dpi='figure', bbox_inches=None)
    plt.show()
