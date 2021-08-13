import numpy as np
from test_algos import *


class PnCDimension:
    def __init__(self, reward_history, lxy, dia):
        self.reward_history = reward_history
        self.lxy = lxy
        self.dia = dia
        self.current_state = [self.lxy, self.dia]

    def get_PnC_dimension(self):
        if self.lxy is not None and self.dia is not None:
            state, action = implement_policy(self.reward_history, self.current_state)
        else:
            state, action = implement_policy(self.reward_history, self.current_state)

        PnC_dimension = state
        return PnC_dimension, action

