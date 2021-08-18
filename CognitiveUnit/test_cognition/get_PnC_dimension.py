import numpy as np
from test_algos import *


class PnCDimension:
    def __init__(self, reward_history, input_state):
        self.reward_history = reward_history
        self.current_state = input_state

    def get_dimension(self):
        state, action = implement_policy(self.reward_history, self.current_state)
        PnC_dimension = state
        action = np.array(action)
        return PnC_dimension, action

