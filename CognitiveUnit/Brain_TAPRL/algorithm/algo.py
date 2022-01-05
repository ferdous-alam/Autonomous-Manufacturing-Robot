import numpy as np
from lib.create_options import CreateOptions
import csv
from visualizations import visualize_samples as vo
from lib.get_reward_from_AMSPnC_data import get_reward_from_AMSPnC_data
from lib.get_reward_from_AMSPnC_data import extract_rewards



def run_TAPRL_feedback(iter_num):

    # append to log file for experiment details
    exp_num = 3  # DO NOT CHANGE, this is fixed!!!
    log_file = open("dump/experiment_no_{}_details.txt".format(exp_num), "a")
    # ---- hyperparameters -------------
    num_of_options = 3
    H = 3
    epsilon = 0.75
    # ----------------------------------

    # ---------------- load source data ----------------------------------
    # load trained Q-table and source reward function
    R_source = np.load('data/source_reward.npy')
    R_source = R_source.T
    # --------------------------------------------------------------------

    # special case: only for the first iteration
    if iter_num == 0:
        """
        If it is the first iteration then 
        choose the predefined initial state 
        for all experiments, x0 = [1, 2] or [1, 6] or [3, 1]
        i.e. x0 = [index_of_dia, index_of_lxy]
        Also, 
            load the source reward distribution as 
                the original source reward from FEM simulated data, 
                the source reward is saved as lxy X d so we need to 
                take the transpose of the source reward each time
            load the trained Q-table using the original FEM source reward
        """
        all_initial_states = []
        options_cache = []
        options_states = []
        all_next_states = []
        option_performance = []

        # get the fixed initial condition for the first iteration
        # initial_state = [3, 1]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        # initial_state = [1, 6]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        initial_state = [1, 2]
        all_initial_states.append(initial_state)
        np.save('dump/all_initial_states.npy', all_initial_states)  # save initial state info
        np.save('dump/options_cache.npy', options_cache)  # save initial options info
        np.save('dump/options_states.npy', options_states)  # save initial options states info
        np.save('dump/all_next_states.npy', all_next_states)  # save all next states
        np.save('dump/option_performance.npy', option_performance)  # save all next states

        # load trained source Q-table
        Q_table = np.load('data/Q_trained_source _original.npy')
        np.save('data/Q_table.npy', Q_table)  # save in a different name to overwrite later

    all_initial_states = np.load('dump/all_initial_states.npy')   # load all initial states
    all_initial_states = all_initial_states.tolist()  # save as a list for appending

    # load created options data, we will pop items from this cache
    # to keep track of the current option; only the current options will be used to
    # find out the current state that will be passed to the AMSPnC machine
    options_cache = np.load('dump/options_cache.npy')
    options_states = np.load('dump/options_states.npy')
    all_next_states = np.load('dump/all_next_states.npy')
    option_performance = np.load('dump/option_performance.npy')   # this gets updated from the PnC spectral response

    # convert all arrays to lists
    options_cache = options_cache.tolist()
    options_states = options_states.tolist()
    all_next_states = all_next_states.tolist()
    option_performance = option_performance.tolist()

    # check whether options cache is null,
    # only create options when the cache is empty
    state = all_initial_states[-1]  # current initial state

    if len(options_cache) == 0:
        if iter_num != 0:
            reward_vals = extract_rewards()   # all rewards values only from the csv files
            option_rewards = reward_vals[len(reward_vals) - num_of_options:]
            option_performance = option_rewards  # only the reward of the last state will be the performance objective
            best_option_idx = np.argmax(option_performance)
            state = all_next_states[best_option_idx]

            # update the log file
            option_details = "         Option selection: -----------------> \n"\
                             "                  Option states: {}, \n "\
                             "                  Option performance: {}, \n"\
                             "                  selected state: {} \n".format(all_next_states, option_performance,
                                                                              state) + "\n" + "\n"
            log_file.write(option_details)



        # create options from current state
        co = CreateOptions(state, H, epsilon, num_of_options)
        options, options_rewards, options_states = co.create_options()
        options_cache = options   # save options to the cache
        all_next_states = [options_states[j][-1] for j in range(len(options_states))]

        # save data for updating
        np.save('dump/options_cache.npy', options_cache)  # override the previous cache of options
        np.save('dump/options_states.npy', options_states)  # override the previous cache of options states
        np.save('dump/all_next_states.npy', all_next_states)  # override all next states from current option

    current_option = options_cache.pop(0)  # current option is a pop item from the options cache
    current_option_states = options_states.pop(0)  # current option is a pop item from the options cache

    # save rest of the options in the cache for future epochs
    np.save('dump/options_cache.npy', options_cache)  # override the previous cache of options
    np.save('dump/options_states.npy', options_states)  # override the previous cache of options

    next_state = current_option_states[-1]  # choose subgoal as the next state
    last_primitive_action = current_option[-1]
    trajectory = [state, last_primitive_action, next_state]
    np.save('dump/current_trajectory.npy', trajectory)
    all_initial_states.append(next_state)  # append to the list of states
    np.save('dump/all_initial_states.npy', all_initial_states)  # save initial state info
    states = all_initial_states
    vo.visualize_samples(iter_num, states)

    # update dump file with relevant information --->
    lxy = np.arange(700, 1100, 50)
    dia = np.arange(350, 650, 50)
    artifact_dimension_prev = [dia[state[0]], lxy[state[1]]]
    artifact_dimension_next = [dia[next_state[0]], lxy[next_state[1]]]
    details = "iteration number: {} ########################## \n"\
              "     Artifact printing step: -----------------> \n"\
              "         current_state: {}, \n "\
              "         current option: {}, \n"\
              "         options cache: {},    \n"\
              "         next states cache: {},    \n" \
              "         next_state: {}, \n"\
              "         artifact_to_be_printed_now: {} micro meters\n" \
              "         artifact_to_be_printed_next: {} micro meters\n".format(
                iter_num + 1, state, current_option, options_cache, all_next_states, next_state,
                artifact_dimension_prev, artifact_dimension_next) + "\n" + "\n"
    log_file.write(details)
    log_file.close()

    # return the indices of the artifact to be printed
    artifact_to_be_printed = artifact_dimension_prev

    return artifact_to_be_printed


def run_TAPRL_update(iter_num):
    exp_num = 3  # DO NOT CHANGE, THIS IS FIXED!!

    # Algo parameters
    alpha = 0.5
    gamma = 0.99

    # 9 possible actions to choose from
    actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
               [-1, 1], [-1, -1], [1, -1], [0, 0]]

    # load real-time reward csv file
    reward = get_reward_from_AMSPnC_data(iter_num)
    trajectory = np.load('dump/current_trajectory.npy')
    trajectory = trajectory.tolist()   # convert to list
    state, last_primitive_action, next_state = trajectory

    # load Q-table
    Q_table = np.load('data/Q_table.npy')

    action_idx = actions.index(last_primitive_action)
    next_best_action_idx = np.argmax(Q_table[next_state[0], next_state[1], :])

    # keep track of Q-value before update
    Q_vals_before = np.copy(Q_table[state[0], state[1], :])

    # update Q-table
    Q_table[state[0], state[1], action_idx] += \
        alpha * (
                reward + gamma * Q_table[next_state[0],
                                         next_state[1],
                                         next_best_action_idx] - Q_table[
                    state[0], state[1], action_idx])
    # keep track of Q-value before update
    Q_vals_after = Q_table[state[0], state[1], :]

    # save Q-table
    np.save('data/Q_table.npy', Q_table)

    # update dump file with relevant information --->
    lxy = np.arange(700, 1100, 50)
    dia = np.arange(350, 650, 50)
    current_state_dimensions = [dia[state[0]], lxy[state[1]]]
    next_state_dimensions = [dia[next_state[0]], lxy[next_state[1]]]

    log_file = open("dump/experiment_no_{}_details.txt".format(exp_num), "a")
    details = "     Brain update step: -----------------> \n" \
              "         reward: {}\n" \
              "         trajectory:---> s_t: {}, a_t: {}, r: {}, s_t+1: {} \n" \
              "         Q[s_t, a_t] before update: {}\n " \
              "        Q[s_t, a_t] after update: {}\n" \
              " ------------------------------------------------------- ".format(
                reward, current_state_dimensions, last_primitive_action,
                reward, next_state_dimensions, Q_vals_before, Q_vals_after) + "\n" + "\n"\

    log_file.write(details)
    log_file.close()

    return None


if __name__ == "__main__":
    for i in range(2):
        artifact_to_be_printed = run_TAPRL_feedback(i)
        run_TAPRL_update(i)



