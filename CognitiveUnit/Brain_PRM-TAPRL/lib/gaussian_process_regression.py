import numpy as np
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from visualizations.plot_GP import plotGPmean


class GaussianProcess:
    """
    Details: Gaussian process is a way supervised algorithm that
    uses gaussian process (not gaussian random variable) to fit
    the training data. It can be used in both regression and
    classification.
      f(x) ~ GP(m(x), k(x,x')); x ---> input feature vector
      GP ---> Gaussian Process
      k(.,.) ---> kernel function/covariance function
    The kernel compares the similarity between the input data,
    various kernel can be chosen according to their characteristics,
    for example, RBF kernel makes smooth prediction
    Here, we use the RBG kernel
         k(x1, x2) = sigma_f^2 * exp(-0.5/length_scale^2 *||x1-x2||)
         ||x1-x2|| = norm/distance

    reference:
    Rasmussen, Carl Edward. ”Gaussian processes in machine learning.”
     Summer School on Machine Learning. Springer, Berlin, Heidelberg, 2003.
    """
    def __init__(self, data, length_scale,
                 sigma_f, sigma_y):
        """

        :param X: training data
        :param Y: training target
        :param X_s: test data
        :param Y_s: test target
        :param length_scale: horizontal scale length of kernel function
        :param sigma_f: vertical scale length of kernel function
        :param sigma_y: variance of noise in the training data
        """
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_y = sigma_y
        self.X = np.array(data[0]).reshape(-1, 1)
        self.Y = np.array(data[1]).reshape(-1, 1)

        # make smooth prediction
        dia_test = np.arange(350, 601, 1.0)
        lxy_test = np.arange(700, 1051, 1.0)
        self.X1_s, self.X2_s = np.meshgrid(dia_test, lxy_test)
        self.X_s = np.c_[self.X1_s.ravel(), self.X2_s.ravel()]

    def kernel(self, X1, X2):
        """

        :param X1: data
        :param X2: data
        :return: K: kernel function
        """
        # if X2 is None:
        #     dists = spdist.pdist(X1, metric='euclidean')
        #     K = self.sigma_f**2 * np.exp(-0.5*dists/self.length_scale)
        #     K = spdist.squareform(K)
        # else:
        #     dists = spdist.cdist(X1, X2, metric='euclidean')
        #     K = self.sigma_f ** 2 * np.exp(-0.5 * dists / self.length_scale)
        norm = np.sum(X1 ** 2, 1).reshape(
            -1, 1) + np.sum(X2 ** 2, 1).reshape(-1, 1) - 2 * np.dot(X1, X2.T)
        K = self.sigma_f ** 2 * np.exp(-0.5 / self.length_scale ** 2 * norm)

        return K

    def posterior(self):
        """
        :return: mu_s = mean of predicted data,
                cov_s = variance of predicted data
        """
        Ky = self.kernel(self.X, self.X) +\
            self.sigma_y ** 2 * np.eye(len(self.X))
        K_s = self.kernel(self.X, self.X_s)
        K_ss = self.kernel(self.X_s, self.X_s)

        mu_s = K_s.T.dot(inv(Ky)).dot(self.Y)
        cov_s = K_ss - K_s.T.dot(inv(Ky)).dot(K_s)

        return mu_s, cov_s

    def plot_gp(self, Y_test):
        plotGPmean(self.X_s, self.Y_s, Y_test, iter_num=1, save_plot=False)



