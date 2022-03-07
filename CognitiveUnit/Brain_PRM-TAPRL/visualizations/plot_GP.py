import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Plot Gaussian Process
def plotGPmean(X1, X2, gpMean, gpStd=None, save_plot=False):
    """
    Plot Gaussian process mean and covariance
    X1: 2 dimensional input
    X2: 2 dimensional input
    gpMean: 2 dimensional mean value for GP prediction
    gpStd: 2 dimensional standard deviation from GP prediction
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, gpMean, linewidth=0.1)

    if gpStd:
        ax.plot_surface(X1, X2, gpMean + gpStd, alpha=0.2, linewidth=0.1)
        ax.plot_surface(X1, X2, gpMean - gpStd, alpha=0.2, linewidth=0.1)
    ax.xaxis.labelpad, ax.yaxis.labelpad, ax.zaxis.labelpad = 20, 20, 20
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('$x_1$', fontsize=20)
    ax.set_ylabel('$x_2$', fontsize=20)
    ax.set_zlabel('Mean value', fontsize=20)
    if save_plot:
        # save plot as pdf
        plt.savefig('figures/plot_gp_mean.pdf', format='pdf', bbox_inches='tight', dpi=1200)

    plt.show()






