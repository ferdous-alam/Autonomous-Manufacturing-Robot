import GPy
import numpy as np
import contextlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# load data
Rs = np.load('../source_reward.npy')
lxy = np.arange(700, 1100, 50)
dia = np.arange(350, 650, 50)

# pre-process data for regression
X1, X2 = np.meshgrid(dia, lxy)
X1, X2, Y_train = X1.flatten(), X2.flatten(), Rs.flatten()
X1, X2, Y_train = X1.reshape(-1, 1), X2.reshape(-1, 1), Y_train.reshape(-1, 1)
X_train = np.concatenate((X1, X2), axis=1)

# define kernel
ker = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
# create simple GP model
m = GPy.models.GPRegression(X_train, Y_train, ker)
m.optimize_restarts(num_restarts=10)

# pre-process test data
lxy_test = np.linspace(350, 600, 1000)
dia_test = np.linspace(700, 1050, 1000)
X1_t, X2_t = np.meshgrid(lxy_test, dia_test)

X1_test, X2_test = X1_t.flatten(), X2_t.flatten()
X1_test, X2_test = X1_test.reshape(-1, 1), X2_test.reshape(-1, 1)
X_test = np.concatenate((X1_test, X2_test), axis=1)

Y_pred, Y_cov = m.predict(X_test)


mean = Y_pred.reshape(len(dia_test), len(lxy_test))
var = Y_cov.reshape(len(dia_test), len(lxy_test))

# plot
plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X1_t, X2_t, mean, cmap='viridis', edgecolor='none')
plt.show()
