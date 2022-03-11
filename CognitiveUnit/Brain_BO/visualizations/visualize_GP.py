import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
import warnings
import matplotlib.pyplot as plt

def plotGP(iter_num):
    warnings.filterwarnings('ignore')#ignore warnings from sklearn


    X_step = np.load('data/suggestions_history.npy')

    R_step = np.load('data/BO_reward_history.npy')

    kernel = 1 * RBF(1, (1e-2, 1e2))
    gp = GPR(kernel=kernel, alpha=.1, n_restarts_optimizer=10, normalize_y=False)
    gp.fit(X_step, R_step)
    lxy_test = np.linspace(350, 600, 1000)
    dia_test = np.linspace(700, 1050, 1000)
    X1_org_test, X2_org_test = np.meshgrid(lxy_test, dia_test)

    X1_test, X2_test = X1_org_test.flatten(), X2_org_test.flatten()
    X1_test, X2_test = X1_test.reshape(-1, 1), X2_test.reshape(-1, 1)
    X_test = np.concatenate((X1_test, X2_test), axis=1)
    R_GP, R_var = gp.predict(X_test, return_std=True)

    R_GP = R_GP.reshape(len(dia_test), len(lxy_test))

    # plot
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X1_org_test, X2_org_test, R_GP, cmap='viridis', edgecolor='none')
    plt.show()


    plt.savefig('figures/GP_iter_{}.pdf'.format(iter_num),
                format='pdf', bbox_inches='tight', dpi=1200)

