import numpy as np
from test_algos import CognitionAlgorithms


class PnCDimension:
    def __init__(self, reward_history, lxy, dia):
        self.reward_history = reward_history
        self.lxy = lxy
        self.dia = dia
        self.current_state = [self.lxy, self.dia]

    def get_PnC_dimension(self):
        if self.lxy is not None and self.dia is not None:
            algo = CognitionAlgorithms()
            state = algo.Q_learning_vanilla(self.reward_history, self.current_state)
        else:
            algo = CognitionAlgorithms()
            state = algo.Q_learning_vanilla(self.reward_history, self.current_state)

        PnC_dimension = state
        return PnC_dimension
