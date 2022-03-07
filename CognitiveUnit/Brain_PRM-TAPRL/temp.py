import numpy as np
from lib.indices_to_artifact_dims import indices_to_artifact_dims
from lib.create_options import CreateOptions
from visualizations import visualize_samples as vo
from lib.get_reward_from_AMSPnC_data import get_reward_from_AMSPnC_data
from lib.get_reward_from_AMSPnC_data import extract_rewards
from algorithm.utils import *


iter_num = 3
exp_num = 3
# -----------------------------
alpha = 0.5
gamma = 0.99
num_of_options = 3  # number of options to be created
H = 5  # length of each option
epsilon = 0.75  # exploration for creating options

# 9 possible actions to choose from
actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
           [-1, 1], [-1, -1], [1, -1], [0, 0]]
# -------------------------------------

# --- load the following files for storing data -----------
options_cache = np.load('dump/options_cache.npy')
options_states_cache = np.load('dump/options_states_cache.npy')
subgoals_cache = np.load('dump/subgoals_cache.npy')
stored_states = np.load('data/stored_states.npy')  # save in a different name to overwrite later

# convert the loaded data into lists for convenience
options_cache = options_cache.tolist()
options_states_cache = options_states_cache.tolist()
subgoals_cache = subgoals_cache.tolist()
stored_states = stored_states.tolist()
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
state = stored_states[-1]  # last state of the stored_states is the current state

if not options_cache:
    # -------------------------------
    # check whether the "options_cache" is empty
    # we need to create a new set of 'm' options and corresponding 'm' subgoals,
    # also need to check whether this is the first iteration, because during the first iteration we
    # need to initialize a lot of things
    # -------------------------------

    if iter_num != 0:
        # this is not the first iteration --> # fit a gaussian process model to
        # the already available reward values from the previous iterations
        reward_vals = extract_rewards()  # extract reward from saved files

        # fit gaussian process model to the dataset with Y = reward_vals, X = states
        # create dataset
        assert len(stored_states) == len(reward_vals), f"dataset input and output size mismatch"
        dataset = [stored_states, reward_vals]
        # fit GP model
        # R_source_current = get_reward_model(dataset)

    # -------------------------------------------------------------------
    # STEP 2: ---> get source reward model (already in 6 x 8 shape, no need to modify)
    # -------------------------------------------------------------------
    R_source = get_source_reward_model(iter_num)

    # -------------------------------------------------------------------
    # STEP 3: ---> get optimal value function by training agent using current source reward model
    # note: the optimal value function is a one dimensional array of 48, convert it to 6x8 shape
    # -------------------------------------------------------------------
    opt_value_func = train_agent(R_source)
    opt_value_func = opt_value_func.reshape(6, 8)

    # create new set of options because cache is empty
    co = CreateOptions(state, H, epsilon, num_of_options, R_source, opt_value_func)
    # options_info is a dictionary with keys 'options', 'options states', 'subgoals states'
    options_info = co.create_options()
    options_cache = options_info['options']
    options_states_cache = options_info['options states']
    subgoals_cache = options_info['subgoals states']
    # if duplicate states are needed to be removed: --->
    # options_states_cache = [list(t) for t in set(tuple(element) for element in options_states_cache)]
    log_file = open("experiment_no_{}_details.txt".format(exp_num), "a")  # open log file
    option_create_details = "          Option creation: -----------------> \n" \
                            "              Options created: {}, \n" \
                            "              Options states: {}, \n" \
                            "              subgoals states: {}".format(options_cache, options_states_cache,
                                                                       subgoals_cache) + "\n" + "\n"
    log_file.write(option_create_details)
    log_file.close()  # close log file

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

