import numpy as np


class PnCMfg:
    def __init__(self, env_type, reward_dist):
        """
        PnC manufacturing environment:
        Environment for PnC manufacturing benchmark study,

        sate-space:
        dimension of state-space = 48,
        where,
        dia = 350, 400, 450, 500, 550, 600 (micrometers)
        lxy = 700, 750, 800, 850, 900, 950, 1000, 1050 (micrometers)
        state = [dia_idx lxy_idx]

        action_space:
        9 possible actions at each state,
        action-space = {up, down, left, right, southeast, southwest, northeast,
                        northwest, stay}

        reward:
        Reward value depends on which environment we are considering,
        to calculate the reward data is loaded based on the environment
        type, this data was created using MATLAB:
            DOE rewards: Design-of-experiment data obtained from the AMSPnC
                    physical machine
            FEM rewards: Data collected from FEM simulation performed in
                        commercially available COMSOL software
            shape of reward_dist: 6 x 8
        """
        self.env_type = env_type
        # import data ---> Add column header
        self.lxy = np.arange(700, 1100, 50)
        self.dia = np.arange(350, 650, 50)
        self.X1, self.X2 = np.meshgrid(self.lxy, self.dia)

        self.reward_dist = reward_dist
        # # choose reward distribution according to environment type if not specified otherwise
        # if env_type == 'source':
        #     self.rewards_dist = np.load('/home/ghost-083/Research/Codes/reward_learning_GP/'
        #                                 'reward_learning_GP/data/source_reward.npy')
        #
        # elif env_type == 'target':
        #     self.rewards_dist = np.load('/home/ghost-083/Research/Codes/reward_learning_GP/'
        #                                 'reward_learning_GP/data/target_reward.npy')
        # else:
        #     raise Exception('Invalid environment type, please choose between '
        #                     'deterministic/model/stochastic')
        # self.rewards_dist = self.rewards_dist.T   # take the transpose for convenience

    def find_new_state(self, state, action):
        """
        Check if the current state is at the boundary,
        if not, then choose the next state and Smake
        the full state, otherwise stay at the previous state
        :rtype: object
        :return: new state
        """

        next_state = [state[0], state[1]]
        next_state[0] = min(max(state[0] + action[0], 0), 5)
        next_state[1] = min(max(state[1] + action[1], 0), 7)

        reward = self.reward_dist[state[0], state[1]]

        return next_state, reward

    def reset(self):
        lxy_idx, dia_idx = np.random.choice(8), np.random.choice(6)
        curr_state = [dia_idx, lxy_idx]
        reward = self.reward_dist[curr_state[0], curr_state[1]]
        return curr_state, reward

    def step(self, state, action):
        return self.find_new_state(state, action)


