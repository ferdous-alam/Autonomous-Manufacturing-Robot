
def gen_waypoints_1(waypoints, i, wh, wv, h_val, L, W):
    if i % 2 != 0:
        waypoints.append([min(i * wh, L), 0])
        if wv <= W:
            waypoints.append([0, min(i * wv, W)])
        else:
            waypoints.append([h_val, min(i * wv, W)])
    else:
        waypoints.append([0, min(i * wv, W)])
        waypoints.append([min(i * wh, L), 0])


def gen_waypoints_2(waypoints, i, wh, wv, L, W, idx_h, ph):
    if i % 2 == 0:
        waypoints.append([ph + idx_h * wh, W])
        waypoints.append([i * wh, 0])
    else:
        waypoints.append([i * wh, 0])
        waypoints.append([ph + idx_h * wh, W])


def gen_waypoints_3(waypoints, i, wh, wv, L, W, idx_h, idx_v, ph, pv):
    if i % 2 == 0:
        waypoints.append([ph + idx_h * wh, i * wv])
        waypoints.append([L, pv + idx_v * wv])
    else:
        waypoints.append([L, pv + idx_v * wv])
        waypoints.append([ph + idx_h * wh, i * wv])


def gen_waypoints_4(waypoints, i, wh, wv, L, W, idx_h, idx_v, ph, pv):
    if i % 2 == 0:
        waypoints.append([ph + idx_h * wh, W])
        waypoints.append([L, pv + idx_v * wv])
    else:
        waypoints.append([L, pv + idx_v * wv])
        waypoints.append([ph + idx_h * wh, L])

