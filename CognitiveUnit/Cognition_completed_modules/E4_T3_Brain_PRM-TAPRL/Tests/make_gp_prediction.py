import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from visualizations.plot_GP import plotGPmean


def make_gp_prediction(num_of_samples, seed):
    np.random.seed(seed)
    # load data -----
    Rs = np.load('../data/source_reward.npy')
    lxy = np.arange(700, 1100, 50)
    dia = np.arange(350, 650, 50)
    X, Y = np.meshgrid(dia, lxy)

    # scikit-learn gaussian process regression -----
    X1, X2, R_train = X.flatten(), Y.flatten(), Rs.flatten()
    X1, X2, R_train = X1.reshape(-1, 1), X2.reshape(-1, 1), R_train.reshape(-1, 1)
    idx_list = [i for i in range(len(X1))]
    idx = list(set(np.random.choice(idx_list, num_of_samples)))
    X1_train = X1[idx]
    X2_train = X2[idx]
    X_train = np.concatenate((X1_train, X2_train), axis=1)
    R_train = R_train[idx]

    # define kernel
    sigma_y = 0.1
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gp = GPR(kernel=kernel, alpha=sigma_y, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10)
    gp.fit(X_train, R_train)

    # predict on test points
    lxy_test = np.linspace(350, 600, 1000)
    dia_test = np.linspace(700, 1050, 1000)
    X1_org_test, X2_org_test = np.meshgrid(lxy_test, dia_test)

    X1_test, X2_test = X1_org_test.flatten(), X2_org_test.flatten()
    X1_test, X2_test = X1_test.reshape(-1, 1), X2_test.reshape(-1, 1)
    X_test = np.concatenate((X1_test, X2_test), axis=1)
    R_GP, R_std = gp.predict(X_test, return_std=True)

    R_GP = R_GP.reshape(len(dia_test), len(lxy_test))

    # plot
    plotGPmean(X1_org_test, X2_org_test, R_GP, save_plot=False)
