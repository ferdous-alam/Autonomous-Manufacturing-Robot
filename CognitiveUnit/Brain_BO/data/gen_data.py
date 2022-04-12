import numpy as np
import matplotlib.pyplot as plt

def gen_data():
    Rs = np.load('data/source_reward.npy')
    lxy = np.arange(700, 1100, 50)
    dia = np.arange(350, 650, 50)

    X, Y = np.meshgrid(dia, lxy)
    return X, Y, Rs, lxy, dia

#plt.figure(figsize=(12, 8))
#ax = plt.axes(projection='3d')
#ax.plot_surface(X, Y, Rs, cmap='viridis', edgecolor='none')
#plt.show()

#plt.contourf(X, Y, Rs)