import numpy as np
from lib.utilities import *


def algo(lxy, theta, W, L):
    # calculate distance between first two waypoints
    theta_rad = theta * np.pi / 180
    wh = lxy / np.sin(theta_rad)
    wv = lxy / np.cos(theta_rad)
    # ---------------------------------

    waypoints = [[0.0, 0.0]]
    idx_h, idx_v = 0, 0
    horiz, vert = True, True
    horiz_new, vert_new = True, True
    ph, pv = 0, 0
    i = 1

    while horiz_new and vert_new:
        if horiz and vert:
            h_val = ((i * wv) - W) / np.tan(theta_rad)
            gen_waypoints_1(waypoints, i, wh, wv, h_val, L, W)
            ph = max(((i + 1) * wv - W) / np.tan(theta_rad), 0)
            pv = max(((i + 1) * wh - L) * np.tan(theta_rad), 0)
            horiz = (i + 1) * wh <= L
            vert = (i + 1) * wv <= W

        elif not vert and horiz:
            if horiz_new and horiz:
                gen_waypoints_2(waypoints, i, wh, wv, L, W, idx_h, ph)
                idx_h += 1
                pv = ((i + 1) * wh - L) * np.tan(theta_rad)
                horiz_new = ph + idx_h * wh <= L
                horiz = (i + 1) * wh <= L
                vert_new = pv + idx_v * wv <= L

        elif not horiz and vert:
            if vert_new and vert:
                gen_waypoints_3(waypoints, i, wh, wv, L, W, idx_h, idx_v, ph, pv)
                idx_v += 1
                vert = (i + 1) * wv <= W
                vert_new = pv + idx_v * wv <= W
                horiz_new = ph + idx_h * wh <= L
                ph = max(((i + 1) * wv - W) / np.tan(theta_rad), 0)

        elif not horiz and not vert:
            if horiz_new and vert_new:
                gen_waypoints_4(waypoints, i, wh, wv, L, W, idx_h, idx_v, ph, pv)
                idx_v += 1
                idx_h += 1
                horiz_new = ph + idx_h * wh <= L
                vert_new = pv + idx_v * wv <= L
        i += 1

    return waypoints

