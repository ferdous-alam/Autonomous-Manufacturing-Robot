from Test_policy.extract_data_from_history import *
from Test_policy.GP_model import *
from algorithm.utils import *


def get_action(trial_num, state):
    r = extract_rewards(trial_num=trial_num)
    x = extract_states(trial_num=trial_num)

    rewards = GP_reward(x, r)
    rewards = rewards.T
    v_func = value_iteration(rewards)
    policy = get_optimal_policy('value_func', v_func, policy_length=1,
                                start_state=state)

    action = policy[0]

    return action
