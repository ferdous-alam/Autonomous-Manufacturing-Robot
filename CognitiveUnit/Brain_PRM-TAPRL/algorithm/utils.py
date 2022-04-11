import numpy as np
from environment.PnCMfg import PnCMfg
from sklearn.gaussian_process import GaussianProcessRegressor
from visualizations.plot_KDE_estimate import plot_kde_estimate
from visualizations.plot_GP import plotGPmean
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def value_iteration(reward_model):
    """
    input:
        reward_model: 6 x 8 reward model

    output:
        value_func: optimal value function from value iteration
    """

    R_s = reward_model.T
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
    # print(f'value iteration converged: delta={Delta}, theta={theta}')
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
    R_source = np.load('data/source_reward.npy')
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
    R_source = R_source.T   # reshape into 6 x 8
    value_func = value_iteration(R_source)

    return value_func


def get_GP_reward_model(data_set, iter_num, viz_model=None):
    """
    This function takes as input the dataset created by the algorithm at each iteration
    and outputs the corresponding GP model of the target reward function

    input:
        data_set: L x 2 shaped list
    output:
        reward model: 6 x 8 shaped array

    Note: The GP posterior is built on larger number of points than the training dataset. This
    is done to ensure smoothness in the posterior prediction, later only the original state-space
    sized data is returned as output as the intermediate values are not really important to make
    a decision.

    """
    # original value of inputs
    x1 = np.arange(700, 1100, 50)
    x2 = np.arange(350, 650, 50)
    X1_org, X2_org = np.meshgrid(x2, x1)
    X1_org_meshed, X2_org_meshed = X1_org.flatten(), X2_org.flatten()
    X1_org_flat, X2_org_flat = X1_org_meshed.reshape(-1, 1), X2_org_meshed.reshape(-1, 1)
    X_org = np.concatenate((X1_org_flat, X2_org_flat), axis=1)

    # train using GP model from the collected dataset
    # extract inputs from dataset, convert to array and reshape
    lxy = np.array([data_set[0][i][1] for i in range(len(data_set[0]))]).reshape(-1, 1)
    dia = np.array([data_set[0][i][0] for i in range(len(data_set[0]))]).reshape(-1, 1)
    Y_train = np.array(data_set[1]).reshape(-1, 1)
    X_train = np.concatenate((dia, lxy), axis=1)

    # ------------- Kernel details ----------------------------------
    # define kernel:
    #    RBF hyperparameters are taken within a range
    #    noise is taken as 0.01 and kept fixed throughout the experiment if not stated otherwise
    #    optimizer is started 10 times from 10 different initial positions to calculate other kernel
    #    hyperparameters automatically
    # ------------------------------------------------------------
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-4, 1e4))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                  normalize_y=False)
    gp.fit(X_train, Y_train)

    # now we will calculate the posterior for arbitrary many points to
    # make smooth prediction
    dia_test = np.arange(350, 601, 1.0)
    lxy_test = np.arange(700, 1051, 1.0)
    X1_org_test, X2_org_test = np.meshgrid(dia_test, lxy_test)

    X1_test, X2_test = X1_org_test.flatten(), X2_org_test.flatten()
    X1_test, X2_test = X1_test.reshape(-1, 1), X2_test.reshape(-1, 1)
    X_test = np.concatenate((X1_test, X2_test), axis=1)
    Y_test_org, Y_var = gp.predict(X_test, return_std=True)
    Y_test = Y_test_org.reshape(len(lxy_test), len(dia_test))

    # only keep the 8x6 data points as the reward model because the rest are trivial
    X_org = X_org.tolist()
    X_test_list = X_test.tolist()
    downsampled_reward = []
    X_temp = []
    Y_test_org_mod = Y_test_org.tolist()
    for i in range(len(X_test_list)):
        if X_test_list[i] in X_org:
            X_temp.append(X_test_list[i])
            downsampled_reward.append(Y_test_org_mod[i][0])

    # reconstruct the dataset of full state-space
    X1_new = np.array(X_temp)[:, 0].reshape(8, 6)
    X2_new = np.array(X_temp)[:, 1].reshape(8, 6)
    reward_model_GP = np.array(downsampled_reward).reshape(8, 6)

    # visualizations
    if viz_model == 'coarse':
        plotGPmean(X1_new, X2_new, reward_model_GP, iter_num, save_plot=True)
    elif viz_model == 'smooth':
        plotGPmean(X1_org_test, X2_org_test, Y_test, iter_num, save_plot=True)
    return reward_model_GP


def modify_dataset_KDE(stored_artifacts, reward_vals):
    print('modifying dataset for KDE calculation . . .')
    unique_artifacts = []
    reward_cache = []

    for k in range(len(stored_artifacts)):
        if stored_artifacts[k] not in unique_artifacts:
            unique_artifacts.append(stored_artifacts[k])
            reward_cache.append([reward_vals[k]])
        else:
            idx = unique_artifacts.index(stored_artifacts[k])
            reward_cache[idx].append(reward_vals[k])

    return unique_artifacts, reward_cache


def get_KDE_estimate(unique_artifacts, rewards_cache, iter_num):
    print('Performing KDE estimate . . . ')

    rewards_kde_mean = []
    rewards_kde_std = []
    rewards_pred = []

    for k in range(len(rewards_cache)):
        y = np.array(rewards_cache[k])
        x = np.linspace(min(y) - 1.0, max(y) + 1.0, 100)

        bandwidths = 10 ** np.linspace(-1, 1, 100)   # for cross validation of the hyperparameter
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=LeaveOneOut())
        grid.fit(x[:, None])
        b = grid.best_params_['bandwidth']
        kde = KernelDensity(bandwidth=b, kernel='gaussian')
        kde.fit(y[:, None])

        # score_samples returns the log of the probability density
        logprob = kde.score_samples(x[:, None])
        y_pred = np.exp(logprob)
        y_pred = y_pred / sum(y_pred)
        prod = y_pred * x
        mean = sum(prod)
        var = sum(x ** 2 * y_pred) - mean ** 2
        sigma = np.sqrt(var)
        rewards_kde_mean.append(mean)
        rewards_kde_std.append(sigma)
        rewards_pred.append(y_pred)
    dataset = [unique_artifacts, rewards_kde_mean]

    # save kde estimate plot
    plot_kde_estimate(rewards_cache, rewards_pred, rewards_kde_mean, rewards_kde_std,
                      iter_num=iter_num, save_plot=True)

    return dataset, rewards_kde_std


def get_best_state(value_func, H, state):
    opt_policy = get_optimal_policy(func_type="value_func", value_func=value_func,
                                    policy_length=H, start_state=state)

    # dummy reward for calculating next_state
    R_s = np.load('data/source_reward.npy')
    R_s = R_s.T
    # instantiate environment with given reward model
    env = PnCMfg('source', R_s)

    for k in range(len(opt_policy)):
        action = opt_policy[k]
        next_state, _ = env.step(state, action)
        state = next_state
    best_state = state
    return best_state


if __name__ == "__main__":
    from visualizations.plot_GP import plotGPmean

    Rs = np.load('../Tests/source_reward.npy')
    lxy = np.arange(700, 1100, 50)
    dia = np.arange(350, 650, 50)
    m, n = np.meshgrid(dia, lxy)
    m = m.flatten().reshape(-1, 1)
    n = n.flatten().reshape(-1, 1)
    m, n = np.squeeze(m).tolist(), np.squeeze(n).tolist()
    Rs = Rs.flatten().reshape(-1, 1)
    Rs = np.squeeze(Rs).tolist()

    X_vals = []
    R_val = []
    for i in range(len(m)):
        if np.random.rand() < 0.25:  # randomly sample data for test purpose
            X_vals.append([[m[i], n[i]]])
            R_val.append(Rs[i])
    X_vals = np.squeeze(X_vals).tolist()
    data_set = [X_vals, R_val]

    reward_model = get_GP_reward_model(data_set, viz_model='smooth')

