import numpy as np
from environment.PnCMfg import PnCMfg


class CreateOptions:
    def __init__(self, state_init, H, epsilon, num_of_options):
        self.state_init = state_init
        self.H = H   # horizon length
        self.epsilon = epsilon
        self.num_of_options = num_of_options
        # action space
        self.actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                        [-1, 1], [-1, -1], [1, -1], [0, 0]]

        # load trained Q-table and source reward function
        R_s = np.load('data/source_reward.npy')
        self.R_source = R_s.T
        self.Q_table = np.load('data/Q_table.npy')
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
            if rand_num <= self.epsilon:
                action_idx = np.random.choice(9)
            else:
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

    def get_subgoal(self):
        option, option_states, option_rewards = self.greedy_option(self.H)
        max_reward_idx = option_rewards.index(max(option_rewards))
        subgoal = option_states[max_reward_idx]
        Delta = max_reward_idx + 1
        return subgoal, Delta

    def create_options(self):
        # identify subgoal state
        subgoal, Delta = self.get_subgoal()
        # initialization
        options_set = {}
        options_rewards_set = {}
        options_states_set = {}

        # get greedy option and append as the first element of the options set
        option_g, option_states_g, option_rewards_g = self.greedy_option(Delta)
        options_set[0] = option_g
        options_rewards_set[0] = option_rewards_g
        options_states_set[0] = option_states_g

        for i in range(self.num_of_options - 1):
            option, option_rewards,  option_states = self.create_option(Delta)
            options_set[i+1] = option
            options_rewards_set[i+1] = option_rewards
            options_states_set[i+1] = option_states

        # convert dictionaries to list
        options = [options_set[i] for i in range(len(options_set))]
        options_rewards = [options_rewards_set[i] for i in range(len(options_rewards_set))]
        options_states = [options_states_set[i] for i in range(len(options_states_set))]

        return options, options_rewards, options_states


if __name__ == "__main__":
    create_options = CreateOptions([2, 4], 3, 0.75, 3)
    options_set, options_rewards_set, options_states_set = create_options.create_options()
    print(options_set)
    # # viz
    # plot_options(options_states_set)




