import numpy as np
import csv
from environment.PnCMfg import PnCMfg
from lib.get_reward_from_AMSPnC_data import get_reward_from_AMSPnC_data
from lib.get_policy import get_policy
from visualizations import visualize_samples as vo


def run_Q_learning_feedback(iter_num):
    """

    :param iter_num:
    :return:
    """

    # append to log file for experiment details
    exp_num = 2  # DO NOT CHANGE, this is fixed!!!
    log_file = open("dump/experiment_no_{}_details.txt".format(exp_num), "a")

    # load trained Q-table and source reward function
    R_source = np.load('data/source_reward.npy')
    R_source = R_source.T

    # exploration factor
    epsilon = 0.10

    if iter_num == 0:
        """
        If it is the first iteration then 
        choose the predefined initial state 
        for all experiments, x0 = [1, 2] 
        i.e. x0 = [index_of_dia, index_of_lxy]
        Also 
            load the source reward distribution as 
                the original source reward from FEM simulated data, 
                the source reward is saved as lxy X d so we need to 
                take the transpose of the source reward each time
            load the trained Q-table using the original FEM source reward
        """
        all_initial_states = []
        # get the fixed initial condition for the first iteration
        # initial_state = [1, 2]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        initial_state = [1, 6]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        # initial_state = [3, 1]
        all_initial_states.append(initial_state)
        Q_table = np.load('data/Q_trained_source.npy')
        np.save('data/Q_table.npy', Q_table)  # save in a different name to overwrite later
        np.save('data/all_initial_states.npy', all_initial_states)  # save initial state info

    all_initial_states = np.load('data/all_initial_states.npy')
    all_initial_states = all_initial_states.tolist()   # save as a list for appending
    state = all_initial_states[-1]

    Q_table = np.load('data/Q_table.npy')  # save in a different name to overwrite later

    # get optimal action from the optimal policy
    env = PnCMfg('source', R_source)
    action = get_policy(Q_table, state, epsilon)
    next_state, _ = env.step(state, action)
    all_initial_states.append(next_state)
    current_trajectory = [state, action, next_state]
    np.save('data/current_trajectory.npy', current_trajectory)  # save initial state info
    np.save('data/all_initial_states.npy', all_initial_states)  # save initial state info
    states = all_initial_states
    # save visualization of option states
    vo.visualize_samples(iter_num, states)

    # update dump file with relevant information --->
    lxy = np.arange(700, 1100, 50)
    dia = np.arange(350, 650, 50)
    artifact_dimension_prev = [dia[state[0]], lxy[state[1]]]
    artifact_dimension_next = [dia[next_state[0]], lxy[next_state[1]]]

    details = "iteration number: {} ########################## \n"\
              "     Artifact printing step: -----------------> \n"\
              "         current_state: {}, \n "\
              "        action_taken: {}, \n"\
              "         next_state: {}, \n"\
              "         artifact_to_be_printed_now: {} micro meters\n" \
              "         artifact_to_be_printed_next: {} micro meters\n".format(
                iter_num + 1, state, action, next_state,
                artifact_dimension_prev, artifact_dimension_next) + "\n" + "\n"
    log_file.write(details)
    log_file.close()

    # return the indices of the artifact to be printed
    artifact_to_be_printed = artifact_dimension_prev

    return artifact_to_be_printed


def run_Q_learning_update(iter_num):
    exp_num = 2  # DO NOT CHANGE, THIS IS FIXED!!

    # Algo parameters
    alpha = 0.5
    gamma = 0.99

    # 9 possible actions to choose from
    actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
               [-1, 1], [-1, -1], [1, -1], [0, 0]]

    # load real-time reward csv file
    reward = get_reward_from_AMSPnC_data(iter_num)

    # load current trajectory and extract state, action, next state info
    current_trajectory = np.load('data/current_trajectory.npy')

    current_state, action, next_state = current_trajectory
    trajectory_info = [current_state, action, next_state, reward]
    action = action.tolist()
    action_idx = actions.index(action)
    # load Q-table
    if iter_num == 0:
        Q_table = np.zeros((6, 8, len(actions)))
        np.save('data/Q_table.npy', Q_table)
    else:
        Q_table = np.load('data/Q_table.npy')

    # save trajectory history for post-processing
    with open('data/trajectory_history.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(trajectory_info)

    # identify next best action
    next_best_action_idx = np.argmax(Q_table[next_state[0], next_state[1], :])

    # keep track of Q-value before update
    Q_vals_before = np.copy(Q_table[current_state[0], current_state[1], :])

    # update Q-table
    Q_table[current_state[0], current_state[1], action_idx] += \
        alpha * (
                reward + gamma * Q_table[next_state[0],
                                         next_state[1],
                                         next_best_action_idx] - Q_table[
            current_state[0], current_state[1], action_idx])

    # keep track of Q-value after update
    Q_vals_after = Q_table[current_state[0], current_state[1], :]

    # update dump file with relevant information --->
    lxy = np.arange(700, 1100, 50)
    dia = np.arange(350, 650, 50)
    current_state_dimensions = [dia[current_state[0]], lxy[current_state[1]]]
    next_state_dimensions = [dia[next_state[0]], lxy[next_state[1]]]

    np.save('data/Q_table.npy', Q_table)  # save Q-table

    log_file = open("dump/experiment_no_{}_details.txt".format(exp_num), "a")
    details = "     Brain update step: -----------------> \n" \
              "         reward: {}\n" \
              "         trajectory:---> s_t: {}, a_t: {}, r: {}, s_t+1: {} \n" \
              "         Q[s_t, a_t] before update: {}\n " \
              "        Q[s_t, a_t] after update: {}\n" \
              " ------------------------------------------------------- ".format(
              reward, current_state_dimensions, action, next_state_dimensions,
        reward, Q_vals_before, Q_vals_after) + "\n" + "\n"\

    log_file.write(details)
    log_file.close()

    return None

