from lib.indices_to_artifact_dims import indices_to_artifact_dims
from lib.create_options import CreateOptions
from visualizations import visualize_samples as vo
from lib.get_reward_from_AMSPnC_data import get_reward_from_AMSPnC_data
from lib.get_reward_from_AMSPnC_data import extract_rewards
from algorithm.utils import *


def run_PRM_TAPRL_feedback(iter_num, trial_num):
    """
    Probabilistic reward modeling with temporal abstraction in physics guided reinforcement learning (PRM-TAPRL)
    -------------------------------------------------------------------------------------------------------------
    Description:
    The main idea of PRM-TAPRL is to use a source model to create options or temporally extended actions (hence
    temporal abstractions) and then use those temporally extended actions to sample some suitable states in the target
    environment. Then the next step is to use the rewards corresponding to those sampled states to build a probabilistic
    model of the target reward function. Here we use Gaussian Process (GP) to build the probabilistic reward model.
    This process is repeated until we run out of allowed physical interactions with the system. One challenge in this
    approach is to have multiple reward values corresponding to the same state (because reward is stochastic) and we do
    not want to put same weight to an outlier reward value which is arbitrarily high or low for some errors in the
    physical system. Thus, we use kernel density estimation (KDE) to identify the probability to each reward value for
    the same state. The mean value from this probability distribution is used as the input data to build the GP reward
    model. Note that KDE has a hyperparameter 'bandwidth' which is calculated using cross validation.

    -----------------------------------------------------------------------------------------

    input:
        iter_num: current iteration number from the physical system

    output:
        artifact_dimension: 1x2 vector --> [filament diameter, filament distance]
        Note that we save required information in this step as well.

    -------------------------------------------------------------------------------------------
    """

    # --------- utilities for file names --------
    # THIS IS FIXED! DO NOT CHANGE
    exp_num = 4
    # ---------------------------------

    if iter_num == 0:
        if trial_num == 1:
            initial_state = [1, 6]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        elif trial_num == 2:
            initial_state = [3, 1]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        else:
            initial_state = [1, 2]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!

        state = initial_state

        # initialize the following for later usages
        stored_states = []
        stored_artifacts = []
        options_cache = []    # initialize a cache for options
        options_states_cache = []   # initialize a cache for all terminal states of the options
        subgoals_cache = []
        options_performance = []   # initialize an empty list of the performances of all current options
        # load trained source Q-table
        Q_table = np.load('data/Q_trained_source_original.npy')
        # save cache files
        # save in dump folder
        np.save('dump/options_cache.npy', options_cache)
        np.save('dump/options_states_cache.npy', options_states_cache)
        np.save('dump/subgoals_cache.npy', subgoals_cache)
        np.save('dump/options_performance.npy', options_performance)
        # save in a different name to overwrite later
        np.save('data/Q_table.npy', Q_table)
        np.save('data/stored_states.npy', stored_states)
        np.save('data/stored_artifacts.npy', stored_artifacts)
        # create an empty csv file to store rewards from the AMSPnC machine
        with open("data/reward_history.csv", "w") as my_empty_csv:
            pass
    else:
        state_cache = np.load('dump/current_state_cache.npy')   # saved state from update module
        state = state_cache.tolist()

    stored_states = np.load('data/stored_states.npy').tolist()
    stored_states.append(state)  # append state to the stored states
    np.save('data/stored_states.npy', stored_states)  # save in a different name to overwrite later

    # convert indices to artifact dimensions
    artifact_dimension = indices_to_artifact_dims(state)
    stored_artifacts = np.load('data/stored_artifacts.npy').tolist()
    stored_artifacts.append(artifact_dimension)  # append state to the stored states
    np.save('data/stored_artifacts.npy', stored_artifacts)  # save in a different name to overwrite later

    # ------- update log file ------------------
    # open log file
    log_file = open("experiment_no_{}_{}_details.txt".format(exp_num, trial_num), "a")
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
    vo.visualize_samples(iter_num, stored_states, save_plot=True)

    return artifact_dimension


def run_PRM_TAPRL_update(iter_num, trial_num):
    """
    Algorithm:
    -------------------------------------------
    input ---> source reward model
               trained Q-table or neural net

    1. Loop T times
    2.      get current state
    3.      Use source reward model to train Q-table/value_function
    4.      create options from trained Q-table
    5.      Implement options to sample states
    6.      Create a dataset of states and corresponding rewards
    7.      Fit a GP model to target reward function
    8.      Update the source reward model with the new GP reward model
    9.      save next state to current_state_cache
    -----------------------------------------------------------------------------

    input:
        iter_num: current iteration number from the physical system

    output:
        None: Note that we save required information in this step as well.
    """
    ##################################################################
    #
    #
    #    Set up everything for running the PRM-TAPRL algorithm
    #
    #
    ##################################################################

    # --------- utilities for file names --------
    # THIS IS FIXED! DO NOT CHANGE
    exp_num = 4
    # ---------------------------------

    # ------ parameters ----------------
    num_of_options = 5   # number of options to be created
    H = 5      # length of each option
    epsilon = 0.75    # exploration for creating options

    # 9 possible actions to choose from
    actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
               [-1, 1], [-1, -1], [1, -1], [0, 0]]
    # -------------------------------------

    # --- load the following files for storing data -----------
    options_cache = np.load('dump/options_cache.npy').tolist()
    options_states_cache = np.load('dump/options_states_cache.npy').tolist()
    subgoals_cache = np.load('dump/subgoals_cache.npy').tolist()
    stored_states = np.load('data/stored_states.npy').tolist()  # save in a different name to overwrite later
    stored_artifacts = np.load('data/stored_artifacts.npy').tolist()  # save in a different name to overwrite later
    # ---------------------------------------

    ###########################################################################
    #
    #
    #  ---------------- Implement PRM-TAPRL algorithm ----------------------
    #
    #
    ###########################################################################

    # -------------------------------------------------------------------
    # STEP 1: ---> get current state from the stored states
    # -------------------------------------------------------------------
    state = stored_states[-1]   # last state of the stored_states is the current state

    if not options_cache:
        # -------------------------------
        # check whether the "options_cache" is empty
        #   1. we need to create a new set of 'm' options and corresponding 'm' subgoals,
        #   2. create the dataset
        #   3. reward_model <-- Fit a GP model to the dataset
        #   4. Save the reward model
        #   Note: also need to check whether this is the first iteration, because during the first iteration
        #   we DO NOT need to fit the GP model as the reward_model is already availble from the original
        #   source task
        # -------------------------------

        if iter_num != 0:
            # this is not the first iteration --> # fit a gaussian process model to
            # the already available reward values from the previous iterations
            reward_vals = extract_rewards()   # extract reward from saved files

            # fit gaussian process model to the dataset with Y = reward_vals, X = states
            # create dataset
            assert len(stored_artifacts) == len(reward_vals), f"dataset input and output size mismatch"

            # modify dataset for kernel density estimation: rearrange data to show only unique states and
            # rewards obtained at each state
            unique_artifacts, rewards_cache = modify_dataset_KDE(stored_artifacts, reward_vals)

            # get KDE estimate
            dataset_post_KDE, rewards_kde_std = get_KDE_estimate(unique_artifacts, rewards_cache, iter_num)
            dataset = dataset_post_KDE
            # save dataset
            np.save(f'dump/dataset_{iter_num}.npy', dataset)

            # update log file
            log_file = open("experiment_no_{}_{}_details.txt".format(exp_num, trial_num), "a")  # open log file
            option_create_details = "\n" + "\n" + "          Performing KDE calculations . . . . .\n" "\n" + "\n"
            log_file.write(option_create_details)
            log_file.close()  # close log file

            # ---- fit GP model ----
            source_reward_current = get_GP_reward_model(dataset, iter_num=iter_num, viz_model='smooth')

            # save reward model
            np.save('data/source_reward_current.npy', source_reward_current)

        # -------------------------------------------------------------------
        # STEP 2: ---> get source reward model (already in 6 x 8 shape, no need to modify)
        # -------------------------------------------------------------------
        R_source = get_source_reward_model(iter_num)    # load the source reward model
        # -------------------------------------------------------------------
        # STEP 3: ---> get optimal value function by training agent using current source reward model
        # note: the optimal value function is a one dimensional array of 48, convert it to 6x8 shape
        # -------------------------------------------------------------------
        opt_value_func = train_agent(R_source)
        opt_value_func = opt_value_func.reshape(6, 8)

        # we need to create options from the best state not the current state
        best_state = get_best_state(opt_value_func, H, state)
        print(f'state:{state} --> best state: {best_state}')
        # move to the best to create new set of options
        state = best_state

        # create new set of options because cache is empty
        co = CreateOptions(state, H, epsilon, num_of_options, R_source, opt_value_func)
        # options_info is a dictionary with keys 'options', 'options states', 'subgoals states'
        options_info = co.create_options()
        options_cache = options_info['options']
        options_states_cache = options_info['options states']
        subgoals_cache = options_info['subgoals states']

        # if duplicate states are needed to be removed: --->
        # options_states_cache = [list(t) for t in set(tuple(element) for element in options_states_cache)]
        log_file = open("experiment_no_{}_details.txt".format(exp_num), "a")   # open log file
        option_create_details = "          Option creation: -----------------> \n"\
                                "              Options created: {}, \n"\
                                "              Options states: {}, \n"\
                                "              subgoals states: {}".format(options_cache, options_states_cache,
                                                                           subgoals_cache) + "\n" + "\n"
        log_file.write(option_create_details)
        log_file.close()   # close log file

    # pick the first element from the cache as the current element
    current_option = options_cache.pop(0)  # pop the first item from the options cache as the current option
    current_subgoal_state = subgoals_cache.pop(0)  # current option is a pop item from the options cache
    next_state = current_subgoal_state  # choose subgoal as the next state
    current_trajectory = [state, current_option, next_state]

    next_artifact_dimension = indices_to_artifact_dims(next_state)

    # load real-time reward csv file
    reward = get_reward_from_AMSPnC_data(iter_num)

    # save data
    np.save('dump/options_cache.npy', options_cache)  # override the previous cache of options
    np.save('dump/options_states_cache.npy', options_states_cache)  # override the previous cache of options
    np.save('dump/subgoals_cache.npy', subgoals_cache)  # override the previous cache of options
    np.save('dump/current_state_cache.npy', next_state)  # save next state as the current state cache

    # --------------- update log file -------------------------------
    log_file = open("experiment_no_{}_details.txt".format(exp_num), "a")
    details = "     Brain update step: -----------------> \n \n" \
              "             trajectory:---> s_t: {}, o_t: {}, r: {}, s_t+1: {} \n \n" \
              "             next state: {} \n" \
              "             artifact to be printed next: {} \n" \
              " ------------------------------------------------------- ".format(
                current_trajectory[0], current_trajectory[1],
                reward, current_trajectory[2],
                next_state, next_artifact_dimension) + "\n" + "\n"\

    log_file.write(details)
    log_file.close()

    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # -------------- save final dataset ----------------------------
    # --------------------------------------------------------------
    # --------------------------------------------------------------
    if iter_num == 23:
        stored_artifacts_latest = np.load('data/stored_artifacts.npy').tolist()  # save in a different name to overwrite later
        # modify dataset for kernel density estimation: rearrange data to show only unique states and
        reward_vals_latest = extract_rewards()  # extract reward from saved files
        # rewards obtained at each state
        unique_artifacts_latest, rewards_cache_latest = modify_dataset_KDE(
            stored_artifacts, reward_vals_latest)

        # get KDE estimate
        dataset_post_KDE_latest, rewards_kde_std_latest = get_KDE_estimate(unique_artifacts_latest,
                                                                           rewards_cache_latest, iter_num)
        dataset_latest = dataset_post_KDE_latest
        # save dataset
        np.save(f'dump/dataset_{iter_num}.npy', dataset_latest)
        r_model = get_GP_reward_model(dataset_latest, iter_num=iter_num, viz_model='smooth')

    return None




