import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Plot Gaussian Process
def plotGPmean(X1, X2, gpMean,  iter_num, gpStd=None, save_plot=False):
    """
    Plot Gaussian process mean and covariance
    X1: 2 dimensional input
    X2: 2 dimensional input
    gpMean: 2 dimensional mean value for GP prediction
    gpStd: 2 dimensional standard deviation from GP prediction
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X1, X2, gpMean, cmap='viridis', edgecolor='none')

    if gpStd:
        ax.plot_surface(X1, X2, gpMean + gpStd, alpha=0.2, cmap='viridis', edgecolor='none')
        ax.plot_surface(X1, X2, gpMean - gpStd, alpha=0.2, cmap='viridis', edgecolor='none')
    ax.xaxis.labelpad, ax.yaxis.labelpad, ax.zaxis.labelpad = 20, 20, 20
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_zlabel('Mean value', fontsize=14)

    ax = fig.add_subplot(1, 2, 2)
    ax.contourf(X2, X1, gpMean)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)

    if save_plot:
        # save plot as pdf
        plt.savefig(f'figures/plot_gp_mean_{iter_num}.pdf', format='pdf', bbox_inches='tight', dpi=1200)

    # plt.show()






