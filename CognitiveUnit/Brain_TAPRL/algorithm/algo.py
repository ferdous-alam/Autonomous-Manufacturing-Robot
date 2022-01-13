import numpy as np
from lib.indices_to_artifact_dims import indices_to_artifact_dims
from lib.create_options import CreateOptions
from visualizations import visualize_samples as vo
from lib.get_reward_from_AMSPnC_data import get_reward_from_AMSPnC_data
from lib.get_reward_from_AMSPnC_data import extract_rewards


def run_TAPRL_feedback(iter_num):

    # --------- utilities for file names --------
    # THIS IS FIXED! DO NOT CHANGE
    exp_num = 3
    # ---------------------------------

    if iter_num == 0:
        # initial_state = [3, 1]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        # initial_state = [1, 6]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        initial_state = [2, 4]
        state = initial_state

        # initialize the following for later usages
        stored_states = []
        options_cache = []    # initialize a cache for options
        options_states_cache = []   # initialize a cache for all terminal states of the options
        running_options_states = []   # initialize the list of all final states from current list of options
        options_performance = []   # initialize an empty list of the performances of all current options
        # load trained source Q-table
        Q_table = np.load('data/Q_trained_source _original.npy')
        stored_states.append(state)  # append state to the stored states
        # save cache files
        np.save('dump/options_cache.npy', options_cache)       # save in dump folder
        np.save('dump/options_states_cache.npy', options_states_cache)  # save in dump folder
        np.save('dump/running_options_states.npy', running_options_states)  # save in dump folder
        np.save('dump/options_performance.npy', options_performance)  # save in dump folder
        np.save('data/Q_table.npy', Q_table)  # save in a different name to overwrite later
        np.save('data/stored_states.npy', stored_states)  # save in a different name to overwrite later
    else:
        state_cache = np.load('dump/current_state_cache.npy')   # saved state from update module
        state = state_cache.tolist()

    stored_states = np.load('data/stored_states.npy')
    stored_states = stored_states.tolist()
    stored_states.append(state)  # append state to the stored states
    np.save('data/stored_states.npy', stored_states)  # save in a different name to overwrite later

    artifact_dimension = indices_to_artifact_dims(state)

    # ------- update log file ------------------
    # open log file
    log_file = open("experiment_no_{}_details.txt".format(exp_num), "a")
    # update the log file
    current_artifact_details = "##### epoch: {} -----------------------------  \n" \
                               "      TAPRL feedback: ------------> \n" \
                               "              current state: {} \n" \
                               "              artifact to be printed now: {} \n".format(iter_num,
                                                                                        state, artifact_dimension)
    log_file.write(current_artifact_details)
    log_file.close()
    # ------------------------------
    # visualize and save visualization
    stored_states = np.load('data/stored_states.npy')
    vo.visualize_samples(iter_num, stored_states)

    return artifact_dimension


def run_TAPRL_update(iter_num):
    # --------- utilities for file names --------
    # THIS IS FIXED! DO NOT CHANGE
    exp_num = 3
    # ---------------------------------

    # ------ parameters ----------------
    alpha = 0.5
    gamma = 0.99
    num_of_options = 3   # number of options to be created
    H = 3      # length of each option
    epsilon = 0.75    # exploration for creating options

    # 9 possible actions to choose from
    actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
               [-1, 1], [-1, -1], [1, -1], [0, 0]]
    # -------------------------------------

    # --- load the following files for storing data -----------
    options_cache = np.load('dump/options_cache.npy')
    options_states_cache = np.load('dump/options_states_cache.npy')
    running_options_states = np.load('dump/running_options_states.npy')
    options_performance = np.load('dump/options_performance.npy')  # save in dump folder
    Q_table = np.load('data/Q_table.npy')  # save in a different name to overwrite later
    stored_states = np.load('data/stored_states.npy')  # save in a different name to overwrite later

    # convert the loaded data into lists for convenience
    options_cache = options_cache.tolist()
    options_states_cache = options_states_cache.tolist()
    running_options_states = running_options_states.tolist()
    options_performance = options_performance.tolist()
    stored_states = stored_states.tolist()
    # ---------------------------------------

    # get current state from the stored states
    state = stored_states[-1]   # last state of the stored_states is the current state

    if not options_cache:
        # if this is first iteration of the options_cache is empty
        # we need to create new set of options
        if iter_num != 0:
            reward_vals = extract_rewards()
            option_rewards = reward_vals[len(reward_vals) - num_of_options:]
            options_performance = option_rewards  # only the reward of the last state will be the performance objective
            best_option_idx = np.argmax(options_performance)
            state = running_options_states[best_option_idx]   # overwrite state information

            # ---- update the log file with option selection ---------------
            option_details = "          Option selection: -----------------> \n"\
                             "              Option states: {}, \n "\
                             "              Option performance: {}, \n"\
                             "              selected state: {} \n".format(running_options_states,
                                                                        options_performance,
                                                                        state) + "\n"

            log_file = open("experiment_no_{}_details.txt".format(exp_num), "a")
            log_file.write(option_details)
            log_file.close()

        # create new set of options because cache is empty
        co = CreateOptions(state, H, epsilon, num_of_options)
        options, options_rewards, options_states = co.create_options()
        options_cache = options
        options_states_cache = [options_states[j][-1] for j in range(len(options_states))]
        # if duplicate states are needed to be removed
        # options_states_cache = [list(t) for t in set(tuple(element) for element in options_states_cache)]
        running_options_states = np.copy(options_states_cache)
        log_file = open("experiment_no_{}_details.txt".format(exp_num), "a")
        option_create_details = "          Option creation: -----------------> \n"\
                                "              Options created: {}, \n"\
                                "              Option states: {},".format(options_cache,
                                                                          running_options_states.tolist()) + "\n" + "\n"
        log_file.write(option_create_details)
        log_file.close()

    # pick the first element from the cache as the current element
    current_option = options_cache.pop(0)  # current option is a pop item from the options cache
    current_option_state = options_states_cache.pop(0)  # current option is a pop item from the options cache
    next_state = current_option_state  # choose subgoal as the next state
    last_primitive_action = current_option[-1]
    current_trajectory = [state, current_option, next_state]

    next_artifact_dimension = indices_to_artifact_dims(next_state)

    # load real-time reward csv file
    reward = get_reward_from_AMSPnC_data(iter_num)
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

    # save data
    np.save('dump/options_cache.npy', options_cache)  # override the previous cache of options
    np.save('dump/options_states_cache.npy', options_states_cache)  # override the previous cache of options
    np.save('dump/running_options_states.npy', running_options_states)  # override the previous cache of options
    np.save('dump/options_performance.npy', options_performance)  # override the previous cache of options

    np.save('data/Q_table.npy', Q_table)  # save Q-table
    np.save('dump/current_state_cache.npy', next_state)  # save next state as the current state cache

    # --------------- update log file -------------------------------
    log_file = open("experiment_no_{}_details.txt".format(exp_num), "a")
    details = "     Brain update step: -----------------> \n \n" \
              "             trajectory:---> s_t: {}, o_t: {}, r: {}, s_t+1: {} \n \n" \
              "             Q[s_t, a_t] before: {}\n " \
              "             Q[s_t, a_t] after: {}\n" \
              "             next state: {} \n" \
              "             artifact to be printed next: {} \n" \
              " ------------------------------------------------------- ".format(
                current_trajectory[0], current_trajectory[1],
                reward, current_trajectory[2], Q_vals_before, Q_vals_after,
                next_state, next_artifact_dimension) + "\n" + "\n"\

    log_file.write(details)
    log_file.close()

    return None

