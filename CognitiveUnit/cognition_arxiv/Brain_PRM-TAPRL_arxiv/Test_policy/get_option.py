from Test_policy.extract_data_from_history import *
from Test_policy.GP_model import *
from algorithm.utils import *


def get_option(trial_num, state):
    r = extract_rewards(trial_num=trial_num)
    x = extract_states(trial_num=trial_num)

    rewards = GP_reward(x, r)
    v_func = value_iteration(rewards)
    option = get_optimal_policy('value_func', v_func, policy_length=3,
                                start_state=state)

    return option


def execute_option(env, state, option):
    for k in range(len(option)):
        action = option[k]
        next_state, _ = env.step(state, action)
        state = next_state
    return state
