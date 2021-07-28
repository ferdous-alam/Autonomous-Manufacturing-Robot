import numpy as np
from test_algos import CognitionAlgorithms


def get_PnC_dimension(reward_history, current_state=None):
    print(current_state)
    algo = CognitionAlgorithms()
    state = algo.Q_learning_vanilla(reward_history, current_state)
    lxy, dia = state
    PnC_dimension = [lxy, dia]

    return PnC_dimension


if __name__ == "__main__":
    reward_history = [1, 2, 3, 4]
    state = [20, 20]
    PnC_dimension = get_PnC_dimension(reward_history)
    print(PnC_dimension)
