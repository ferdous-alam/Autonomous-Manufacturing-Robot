import numpy as np
import csv
from environment.PnCMfg import PnCMfg
from lib.get_policy import get_policy
from lib.indices_to_artifact_dims import indices_to_artifact_dims
from visualizations import visualize_samples as vo
from lib.get_reward_from_AMSPnC_data import get_reward_from_AMSPnC_data
from lib.get_reward_from_AMSPnC_data import extract_rewards


class CreateOptions:
    def __init__(self, state_init, H, trial_num=None):
        self.state_init = state_init
        self.H = H   # horizon length
        self.trial_num = trial_num
        # action space
        self.actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                        [-1, 1], [-1, -1], [1, -1], [0, 0]]

        # load trained Q-table and source reward function
        R_s = np.load('data/source_reward.npy')
        self.R_source = R_s.T
        if self.trial_num == 1:
            self.Q_table = np.load('Test_policy/Q_table_trained_E3_T1.npy')
        elif self.trial_num == 2:
            self.Q_table = np.load('Test_policy/Q_table_trained_E3_T2.npy')
        else:
            self.Q_table = np.load('Test_policy/Q_table_trained_E3_T3.npy')
        # instantiate environment
        self.env = PnCMfg('source', self.R_source)

    def create_option(self, Delta):
        # create options
        option = []
        option_states = []
        option_rewards = []
        state = self.state_init

        for i in range(Delta):
            rand_num = np.random.rand()
            action_idx = np.argmax(self.Q_table[state[0], state[1], :])
            action = self.actions[action_idx]
            next_state, reward = self.env.step(state, action)
            # append to the option
            option.append(action)
            option_states.append(state)
            option_rewards.append(reward)
            # update state
            state = next_state
        option_states.append(state)
        return option, option_rewards, option_states

    def greedy_option(self, Delta):
        # create options
        option = []
        option_states = []
        option_rewards = []
        state = self.state_init

        for i in range(Delta):
            action_idx = np.argmax(self.Q_table[state[0], state[1], :])
            action = self.actions[action_idx]
            next_state, reward = self.env.step(state, action)
            # append to the option
            option.append(action)
            option_states.append(state)
            option_rewards.append(reward)
            # update state
            state = next_state
        option_states.append(state)

        return option, option_states, option_rewards

    def create_options(self):
        option, option_states, option_rewards = self.greedy_option(self.H)

        return option, option_states, option_rewards


if __name__ == "__main__":
    create_options = CreateOptions([2, 5], 3, exp_num=2)
    option, option_states, option_rewards = create_options.create_options()
    print(option_states)





