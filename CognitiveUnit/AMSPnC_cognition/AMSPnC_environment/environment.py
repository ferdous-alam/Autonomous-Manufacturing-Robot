import numpy as np


class AMSPnCEnv:
    def __init__(self):
        """
        Environment for producing next state based on current state and current action
        Each state represents a PnC sample,
        |state-space| = 68 x 68
        |action-space| = 9

        action-space = [UP, DOWN, LEFT, RIGHT, SE, SW, NE, NW, no-move]
                    UP = [0, 1]
                    DOWN = [1, 0]
                    LEFT = []
                    RIGHT = []
                    SE = []
                    SW = []
                    NE = []
                    NW = []
                    no-move = []

        Arguments:
            Input:
                state ---> list of state variables i.e. x = [lxy d]
                action -->  list of state variables i.e. a = [UP Down]

            Output:
                next_state --> list of state variables


        """

    def restart(self):
        lb, ub = 0, 68
        state = [np.random.randint(lb, ub), np.random.randint(lb, ub)]

        return state

    def step(self, state, action):
        next_state = [state[i] + action[i] for i in range(len(state))]

        return next_state


if __name__ == '__main__':
    env = AMSPnCEnv()
    # check random state
    random_initial_state = env.restart()
    print(random_initial_state)
    # check next action
    state = random_initial_state
    action = [0, 1]
    next_state = env.step(state, action)
    print(next_state)
