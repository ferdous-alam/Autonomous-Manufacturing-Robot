import numpy as np
from environment.PnCMfg import PnCMfg
from lib.get_optimal_policy import get_optimal_policy
from visualizations import visualize_samples as vo


def run_offline_feedback(iter_num):
    """

    :param iter_num:
    :return:
    """

    # append to log file for experiment details
    exp_num = 1  # DO NOT CHANGE, this is fixed!!!
    log_file = open("dump/experiment_no_{}_details.txt".format(exp_num), "a")

    # load trained Q-table and source reward function
    R_source = np.load('data/source_reward.npy')
    R_source = R_source.T
    Q_table = np.load('data/Q_trained_source.npy')

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
        initial_state = [1, 2]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        all_initial_states.append(initial_state)
        np.save('data/all_initial_states.npy', all_initial_states)  # save initial state info

    all_initial_states = np.load('data/all_initial_states.npy')
    all_initial_states = all_initial_states.tolist()   # save as a list for appending
    state = all_initial_states[-1]

    # get optimal action from the optimal policy
    env = PnCMfg('source', R_source)
    opt_action = get_optimal_policy(Q_table, state)
    next_state, _ = env.step(state, opt_action)
    all_initial_states.append(next_state)
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
              "         starting_state: {}, \n "\
              "         action_taken: {}, \n"\
              "         artifact_indices: {}, \n"\
              "         artifact_to_be_printed_now: {} micro meters\n" \
              "         artifact_to_be_printed_next: {} micro meters\n".format(
                iter_num + 1, state, opt_action, next_state,
                artifact_dimension_prev, artifact_dimension_next) + "\n" + "\n"
    log_file.write(details)
    log_file.close()

    # return the indices of the artifact to be printed
    artifact_to_be_printed = artifact_dimension_prev

    return artifact_to_be_printed


def run_offline_update(iter_num):
    exp_num = 1  # DO NOT CHANGE, THIS IS FIXED!!
    log_file = open("dump/experiment_no_{}_details.txt".format(exp_num), "a")
    details = "     Brain update step: -----------------> \n"\
              "         No update required: offline execution of source optimal policy \n " \
              " ------------------------------------------------------- " + "\n" + "\n"\

    log_file.write(details)
    log_file.close()

    return None