from get_PnC_dimension import PnCDimension


def get_lxy(reward_history, lxy_prev=None, dia_prev=None):
    PnC = PnCDimension(reward_history, lxy_prev, dia_prev)
    PnC_dim = PnC.get_PnC_dimension()
    lxy = PnC_dim[0]
    return lxy


def get_dia(reward_history, lxy_prev=None, dia_prev=None):
    PnC = PnCDimension(reward_history, lxy_prev, dia_prev)
    PnC_dim = PnC.get_PnC_dimension()
    dia = PnC_dim[1]
    return dia


if __name__ == "__main__":
    reward_history = [1, 2, 4]
    lxy = get_lxy(reward_history, 1, 2)
    dia = get_dia(reward_history, 1, 2)
    print(lxy)
    print(dia)
