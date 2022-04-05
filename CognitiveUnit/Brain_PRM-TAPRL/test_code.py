# from Test_policy.get_action import *
# from Test_policy.extract_data_from_history import *
# from Test_policy.GP_model import *
# from algorithm.utils import *
# # action, rewards = get_action(trial_num=1, state=[1, 2])
#
# r = extract_rewards(trial_num=1)
# x = extract_states(trial_num=1)
#
# rewards = GP_reward(x, r)
#
# # instantiate environment with given reward model
# env = PnCMfg('source', rewards.T)
#
# x = np.arange(0, 6, 1)
# y = np.arange(0, 8, 1)
# X, Y = np.meshgrid(x, y)
# states = []
# for i in range(len(x)):
#     for j in range(len(y)):
#         state = [X[j][i], Y[j][i]]
#         states.append(state)
#
# # all possible actions
# actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
#            [-1, 1], [-1, -1], [1, -1], [0, 0]]
#
# # -------- initialize value function -----
# value_func = np.zeros(len(states))
# gamma = 0.98  # discount factor
# theta = 1e-5  # initialize threshold theta to random value
# Delta = 1e5
#
# while Delta > theta:
#     Delta = 0
#     for i in range(len(states)):
#         state = states[i]
#         v = value_func[i]
#         val_cache = []
#         actions_cache = []
#         for action in actions:
#             next_state, reward = env.step(state, action)  # deterministic transition
#             j = states.index(next_state)
#             val = np.sum((reward + gamma * value_func[j]))
#             val_cache.append(val)
#             actions_cache.append(action)
#
#         max_V_idx = val_cache.index(max(val_cache))
#         value_func[i] = val_cache[max_V_idx]
#         Delta = max(Delta, abs(v - value_func[i]))
#
# policy = get_optimal_policy('value_func', value_func, policy_length=1,
#                             start_state=[1, 6])
#
# action = policy[0]
