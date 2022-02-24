import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from celluloid import Camera
import matplotlib.animation as animation


# input variables
theta = 45
lxy = 0.75  # PnC unit cell spacing (mm)
d = 0.5  # PnC unit cell spacing (mm)
# PnC filament orientation (degrees)
L = 34.0  # PnC length (mm)
W = 34.0  # PnC width (mm)
theta_rad = theta * np.pi / 180
wh = lxy / np.sin(theta_rad)
wv = lxy / np.cos(theta_rad)

wp = np.load('waypoints_theta_{}.npy'.format(theta))
xs = [wp[i][0] for i in range(len(wp))]
ys = [wp[i][1] for i in range(len(wp))]
zs = [0] * len(xs)
#
# # camera setting
# camera = Camera(plt.figure())
# # ----------- plot figure --------
# x_all, y_all = [], []
# x_o = np.linspace(0, 34, 100)
# for i in range(len(xs)):
#     y_o = i * wv - (wv / wh) * x_o
#     x_all.append(x_o)
#     y_all.append(y_o)
#
# for i in range(len(xs)):
#     # for j in range(len(xs)):
#     #     plt.plot(x_all[j], y_all[j], '--', color='black', alpha=0.10)
#     plt.scatter(xs[:i], ys[:i], c='k', s=100, alpha=0.5)
#     plt.plot(xs[:i], ys[:i], c='k', lw=3.0)
#     plt.xlim([-2, 36])
#     plt.ylim([-2, 36])
#     plt.title(r'$lxy = {}mm, d ={}mm, \theta = {}^0$'.format(lxy, d, theta))
#     camera.snap()
# anim = camera.animate(blit=True)
# anim.save('anim_theta_{}.gif'.format(theta))  # save animation

#
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xs, ys, zs, marker='o')
ax.plot3D(xs, ys, zs, 'gray')
plt.show()

# # plt.plot(xs, ys)
# # plt.show()
