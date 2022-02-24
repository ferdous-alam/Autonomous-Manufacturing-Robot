import numpy as np
from lib.algo import algo
from lib.plot_waypoints import plot_waypoints

# -------------------------------------
# input variables
theta = 90
lxy = 3.0  # PnC unit cell spacing (mm)
# PnC filament orientation (degrees)
L = 34.0  # PnC length (mm)
W = 34.0  # PnC width (mm)

waypoints = algo(lxy, theta, W, L)
plot_waypoints(waypoints, lxy, theta, L, W)
# np.save('lib/waypoints_theta_{}.npy'.format(theta), waypoints)

