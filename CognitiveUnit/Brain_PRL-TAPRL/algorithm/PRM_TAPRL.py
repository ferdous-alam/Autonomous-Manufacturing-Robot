import numpy as np
import GPy
from lib import create_temporal_abstractions as cta
from visualizations import visualize_options as vo


def run_initial_PRM_TAPRL(iter_num, algo_params):
    """

    :param iter_num:
    :param algo_params:
    :return:
    """
    # algo parameters --->
    num_of_options, option_epsilon, H = algo_params
    # num_of_options: number of options to be created
    # option_epsilon: exploration factor for creating options
    # H = 5: horizon length

    # append to log file for experiment details
    exp_num = 1  # DO NOT CHANGE, this is fixed!!!
    log_file = open("dump/experiment_no_{}_details.txt".format(exp_num), "a")

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

        initial_state = [1, 2]  # s = [d, lxy]
        R_source = np.load('data/source_reward.npy')
        R_source = R_source.T
        Q_table = np.load('data/Q_trained_source.npy')
    else:
        all_initial_states = np.load('data/initial_states.npy')
        initial_state = all_initial_states[-1]
        R_source = np.load('data/target_reward_GP_{}.npy'.format(iter_num))
        R_source = R_source.T
        Q_table = np.load('data/Q_trained_source_GP_{}.npy'.format(iter_num))

    options, option_states = cta.create_temporal_abstraction(
        Q_table, R_source, initial_state,
        H, num_of_options, option_epsilon)

    # save visualization of option states
    vo.visualize_option_states(option_states, iter_num)

    option_terminal_states = [option_states[i][-1] for i in range(len(option_states))]

    # update dump file with relevant information --->
    details = "iteration number: {} ########################## \n"\
              "     Artifacts printing step: -----------------> \n"\
              "         starting_state: {}, \n "\
              "         options_created: {} \n "\
              "         option_epsilon: {}, horizon: {}, \n " \
              "         artifacts_to_be_printed: {} \n".format(iter_num + 1,
                                                 initial_state,
                                                 num_of_options,
                                                 option_epsilon,
                                                 H,
                                                 option_terminal_states) + "\n" + "\n"
    log_file.write(details)
    log_file.close()

    # return the indices of the artifacts to be printed
    artifact_indices = option_terminal_states

    return artifact_indices

