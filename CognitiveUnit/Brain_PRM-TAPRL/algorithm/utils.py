import numpy as np
from environment.PnCMfg import PnCMfg
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def value_iteration(reward_model):
    """
    input:
        reward_model: 6 x 8 reward model

    output:
        value_func: optimal value function from value iteration
    """

    R_s = np.load('data/source_reward.npy')
    R_s = R_s.T
    # instantiate environment with given reward model
    env = PnCMfg('source', R_s)

    x = np.arange(0, 6, 1)
    y = np.arange(0, 8, 1)
    X, Y = np.meshgrid(x, y)
    states = []
    for i in range(len(x)):
        for j in range(len(y)):
            state = [X[j][i], Y[j][i]]
            states.append(state)

    # all possible actions
    actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
               [-1, 1], [-1, -1], [1, -1], [0, 0]]

    # -------- initialize value function -----
    value_func = np.zeros(len(states))
    gamma = 0.98  # discount factor
    theta = 1e-5  # initialize threshold theta to random value
    Delta = 1e5

    while Delta > theta:
        Delta = 0
        for i in range(len(states)):
            state = states[i]
            v = value_func[i]
            val_cache = []
            actions_cache = []
            for action in actions:
                next_state, reward = env.step(state, action)  # deterministic transition
                j = states.index(next_state)
                val = np.sum((reward + gamma * value_func[j]))
                val_cache.append(val)
                actions_cache.append(action)

            max_V_idx = val_cache.index(max(val_cache))
            value_func[i] = val_cache[max_V_idx]
            Delta = max(Delta, abs(v - value_func[i]))
    np.save('data/optimal_value_func.npy', value_func)
    print(f'value iteration converged: delta={Delta}, theta={theta}')
    return value_func


def get_optimal_policy(func_type, value_func, policy_length, start_state):
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
    # convert value function to 2D arrays
    value_func = value_func.reshape(6, 8)
    # 9 possible actions to choose from
    actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
               [-1, 1], [-1, -1], [1, -1], [0, 0]]
    x = start_state

    # instantiate environment
    # instantiate environment
    R_source = np.load('../data/source_reward.npy')
    R_source = R_source.T
    env = PnCMfg('source', R_source)
    # reward is a dummy input as it is not important to find the optimal policy
    # rather only to instantiate the environment
    optimal_policy = []

    for i in range(policy_length):
        if func_type == "Q_func":
            action_idx = np.argmax(value_func[x[0], x[1], :])
            opt_action = actions[action_idx]
            x_next, _ = env.step(x, opt_action)
            best_next_state = x_next
        elif func_type == "value_func":
            V_opt_cache = []
            actions_cache = []
            next_states_cache = []
            for action in actions:
                x_next, _ = env.step(x, action)
                v_val = value_func[x_next[0], x_next[1]]
                V_opt_cache.append(v_val)
                actions_cache.append(action)
                next_states_cache.append(x_next)
            opt_val_idx = np.argmax(V_opt_cache)
            best_next_state = next_states_cache[opt_val_idx]
            opt_action = actions_cache[opt_val_idx]  # find the best action
        else:
            raise Exception('choose value function value_func or action value function Q_func')

        optimal_policy.append(opt_action)
        x = best_next_state

    return optimal_policy


def get_source_reward_model(iter_num):
    """

    Calculates the source reward model using the current iteration number

    input:
        iter_num: current iteration number
    output:
        R_s: source reward model

    """

    if iter_num == 0:
        # this is the first iteration, so we will use the original source reward model
        source_reward_model = np.load('data/source_reward.npy')
    else:
        # otherwise, load the current model of the source reward
        source_reward_model = np.load('data/source_reward_current.npy')

    R_source = source_reward_model.T

    return R_source


def train_agent(R_source):
    """
    The agent is trained using any RL algorithm or Dynamic programming
    and the optimal Q-values are returned, we instantiate the environment
    with the current reward model
    input:
        R_source: current model of the source reward
    output:
        Q_trained: trained Q-values --> optimal Q-values

    Note:
        it may be possible to use any policy based method to directly
        return the optimal policy
    """
    # Here we will use dynamic programming to train the agent as the state-space
    # is relatively small.
    value_func = value_iteration(R_source)

    return value_func


def get_GP_reward_model(dataset):

    # train using GP model
    # extract inputs from dataset
    lxy = np.array([dataset[0][i][1] for i in range(len(dataset[0]))]).reshape(-1, 1)
    dia = np.array([dataset[0][i][0] for i in range(len(dataset[0]))]).reshape(-1, 1)
    Y_train = np.array(dataset[1]).reshape(-1, 1)
    X_train = np.concatenate((lxy, dia), axis=1)

    # define kernel
    kernel = 1 * RBF(10, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.1, n_restarts_optimizer=10)
    gp.fit(X_train, Y_train)

    # now we will calculate the posterior for arbitrary many points to
    # make smooth prediction
    dia_test = np.linspace(350, 600, 1000)
    lxy_test = np.linspace(700, 1050, 1000)
    X1_org_test, X2_org_test = np.meshgrid(dia_test, lxy_test)

    X1_test, X2_test = X1_org_test.flatten(), X2_org_test.flatten()
    X1_test, X2_test = X1_test.reshape(-1, 1), X2_test.reshape(-1, 1)
    X_test = np.concatenate((X1_test, X2_test), axis=1)
    Y_test, _ = gp.predict(X_test, return_std=True)
    Y_test = Y_test.reshape(len(lxy_test), len(dia_test))

    # only keep the 6x8 datapoints as the reward model as the rest
    # are not important
    reward_model = Y_test
    return reward_model


if __name__ == "__main__":
    R_s = get_source_reward_model(0)
    V = value_iteration(R_s)
    opt_policy = get_optimal_policy('value_func', V, 10, [1, 2])

