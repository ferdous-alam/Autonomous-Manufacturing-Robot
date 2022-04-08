import numpy as np
from lib.utils import *
import matplotlib.pyplot as plt


state = [3, 1]
R_source = np.load('data/source_reward.npy')
# R_source = R_source.T   # convert to 6 x 8 shape
opt_val_func = value_iteration(R_source)
opt_policy = get_optimal_policy(func_type='value_func', value_func=opt_val_func,
                                policy_length=5, start_state=state)


plt.scatter(state[1], state[0])
for i in range(len(opt_policy)):
    next_state1 = state[0] + opt_policy[i][0]
    next_state2 = state[1] + opt_policy[i][1]
    next_state = [next_state1, next_state2]
    print(state)
    plt.scatter(next_state2, next_state1)
    state = next_state

plt.xlim([0, 7])
plt.ylim([0, 5])
plt.show()
