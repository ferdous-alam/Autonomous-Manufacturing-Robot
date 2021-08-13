from get_PnC_dimension import PnCDimension


def get_lxy_action(reward_history, lxy_prev=None, dia_prev=None):
    PnC = PnCDimension(reward_history, lxy_prev, dia_prev)
    PnC_dim, action = PnC.get_PnC_dimension()
    lxy_action = float(action[0])
    return lxy_action


def get_dia_action(reward_history, lxy_prev=None, dia_prev=None):
    PnC = PnCDimension(reward_history, lxy_prev, dia_prev)
    PnC_dim, action = PnC.get_PnC_dimension()
    dia_action = float(action[1])

    return dia_action
