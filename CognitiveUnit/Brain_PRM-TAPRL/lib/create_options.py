import numpy as np
from environment.PnCMfg import PnCMfg


class CreateOptions:
    def __init__(self, state_init, H, epsilon, num_of_options, R_source, opt_value_func):
        self.state_init = state_init
        self.H = H   # horizon length
        self.epsilon = epsilon
        self.num_of_options = num_of_options
        self.R_source = R_source
        self.opt_value_func = opt_value_func
        # action space
        self.actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                        [-1, 1], [-1, -1], [1, -1], [0, 0]]

        # instantiate environment
        self.env = PnCMfg('source', self.R_source)

    def get_subgoal(self, policy, policy_states):
        subgoal = policy_states[-1]

        return subgoal

    def get_policy(self, exploration_factor):
        """
        Extract the optimal policy from a given Q-function, i.e. Q-table in tabular case
        trained Q-network in neural network
        input:
            Q-function: |S| X |A| dimensional Q-table
                    or, value_func: |S| dimensional value function
            H: length of optimal policy
        output:
            optimal_policy: optimal policy upto fixed horizon of H

        note:
            we need to convert optimal value function into 2D array of size 6 x 8
        """
        # here we use value function to create options
        func_type = 'value_func'
        # instantiate environment
        # instantiate environment
        env = PnCMfg('source', self.R_source)
        # reward is a dummy input as it is not important to find the optimal policy
        # rather only to instantiate the environment
        x = self.state_init  # initial state
        policy = []
        policy_states = [x]
        subgoal = []

        for i in range(self.H):
            if func_type == "Q_func":
                if exploration_factor < np.random.rand():   # greedy action
                    action_idx = np.argmax(self.opt_value_func[x[0], x[1], :])
                else:
                    action_idx = np.random.choice(len(self.actions))  # exploratory action
                opt_action = self.actions[action_idx]
                x_next, _ = env.step(x, opt_action)
                best_next_state = x_next

            elif func_type == "value_func":
                V_opt_cache = []
                actions_cache = []
                next_states_cache = []
                for action in self.actions:
                    x_next, _ = env.step(x, action)
                    v_val = self.opt_value_func[x_next[0], x_next[1]]
                    V_opt_cache.append(v_val)
                    actions_cache.append(action)
                    next_states_cache.append(x_next)
                if exploration_factor < np.random.rand():   # greedy action
                    opt_val_idx = np.argmax(V_opt_cache)
                else:                                 # exploratory action
                    opt_val_idx = np.random.choice(len(self.actions))
                best_next_state = next_states_cache[opt_val_idx]
                opt_action = actions_cache[opt_val_idx]  # find the best action
            else:
                raise Exception('choose value function value_func or action value function Q_func')

            policy.append(opt_action)
            x = best_next_state
            policy_states.append(x)  # save next state to policy state

        subgoal = self.get_subgoal(policy, policy_states)

        return policy, policy_states, subgoal

    def create_options(self):

        # initialize
        options_set = []
        options_states_set = []
        subgoal_states = []

        # at first append greedy option to options set by selecting exploration = 0.0
        exploration_factor = 0
        greedy_option, policy_states, subgoal = self.get_policy(exploration_factor)
        options_set.append(greedy_option)
        options_states_set.append(policy_states)
        subgoal_states.append(subgoal)

        # for other options set exploration to epsilon value
        for i in range(self.num_of_options - 1):
            option, option_states, subgoal = self.get_policy(self.epsilon)
            options_set.append(option)
            options_states_set.append(option_states)
            subgoal_states.append(subgoal)

        options_info = {'options': options_set,
                        'options states': options_states_set,
                        'subgoals states': subgoal_states}

        return options_info


if __name__ == "__main__":
    from algorithm.utils import *
    R_source = get_source_reward_model(0)
    opt_value_func = train_agent(R_source)
    co = CreateOptions([1, 2], 2, 0.5, 3, R_source, opt_value_func)
    options_info = co.create_options()






