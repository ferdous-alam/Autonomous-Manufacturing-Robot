import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from visualizations.plot_GP import plotGPmean
from sklearn.gaussian_process.kernels import RBF


def GP_reward(X, R):
    kernel = 1.0 * RBF(10, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.01, n_restarts_optimizer=10)
    gp.fit(X, R)

    # original value of inputs
    x1 = np.arange(700, 1100, 50)
    x2 = np.arange(350, 650, 50)
    X1_org, X2_org = np.meshgrid(x2, x1)
    X1_org_meshed, X2_org_meshed = X1_org.flatten(), X2_org.flatten()
    X1_org_flat, X2_org_flat = X1_org_meshed.reshape(-1, 1), X2_org_meshed.reshape(-1, 1)
    X_org = np.concatenate((X1_org_flat, X2_org_flat), axis=1)

    # now we will calculate the posterior for arbitrary many points to
    # make smooth prediction
    dia_test = np.arange(350, 601, 1.0)
    lxy_test = np.arange(700, 1051, 1.0)
    X1_org_test, X2_org_test = np.meshgrid(dia_test, lxy_test)

    X1_test, X2_test = X1_org_test.flatten(), X2_org_test.flatten()
    X1_test, X2_test = X1_test.reshape(-1, 1), X2_test.reshape(-1, 1)
    X_test = np.concatenate((X1_test, X2_test), axis=1)
    Y_test_org, Y_var = gp.predict(X_test, return_std=True)
    Y_test = Y_test_org.reshape(len(lxy_test), len(dia_test))

    # # only keep the 8x6 data points as the reward model because the rest are trivial
    X_org = X_org.tolist()
    X_test_list = X_test.tolist()
    downsampled_reward = []
    X_temp = []
    Y_test_org_mod = Y_test_org.tolist()
    for i in range(len(X_test_list)):
        if X_test_list[i] in X_org:
            X_temp.append(X_test_list[i])
            downsampled_reward.append(Y_test_org_mod[i])

    # reconstruct the dataset of full state-space
    X1_new = np.array(X_temp)[:, 0].reshape(8, 6)
    X2_new = np.array(X_temp)[:, 1].reshape(8, 6)
    reward_model_GP = np.array(downsampled_reward).reshape(8, 6)

    # --- plot GP model --->
    # plotGPmean(X1_org_test, X2_org_test, Y_test, iter_num=1)
    # plotGPmean(X1_new, X2_new, reward_model_GP, iter_num=1)

    return reward_model_GP
