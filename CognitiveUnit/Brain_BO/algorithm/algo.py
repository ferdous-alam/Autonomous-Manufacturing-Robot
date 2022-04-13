import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings, time
from algorithm.acquisition import UCB
from visualizations import visualize_samples as vo
import csv
#from data.gen_data import gen_data


def run_feedback(iter_num, trial_num, iter_of_convergence=24):
    exp_num = 5  # DO NOT CHANGE, this is fixed!!!
    # to help with indexing
    lxy = np.arange(700, 1100, 50).tolist()
    dia = np.arange(350, 650, 50).tolist()
    X, Y = np.meshgrid(dia, lxy)

    if iter_num == 0:  # return initial condition to be printed as first iteration

        all_initial_states = []
        # get the fixed initial condition for the first iteration
        if trial_num == 1:
            initial_state = [3, 1]
        elif trial_num == 2:
            initial_state = [1, 6]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        else:
            initial_state = [1, 2]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!

        # initialize the following for later usages
        stored_states = []
        stored_artifacts = []
        # save in dump folder
        np.save('data/stored_states.npy', stored_states)
        np.save('data/stored_artifacts.npy', stored_artifacts)

        # find true values corresponding to IC indices
        dia_val = dia[initial_state[0]]
        lxy_val = lxy[initial_state[1]]

        X_step = np.array([[dia_val, lxy_val]])
        artifact_to_be_printed = X_step

        # create an empty csv file to store rewards from the AMSPnC machine
        with open("data/reward_history.csv", "w") as my_empty_csv:
            pass

    else:

        # load previously sampled data and associated rewards,
        # initialize gaussian process and call acquisition function to find maximum UCB value
        R_step = np.load('data/BO_reward_history.npy')  # remove when not simulating

        # obtain current sample data
        X_step = np.load('data/suggestions_history.npy')

        # Initialize gaussian process
        warnings.filterwarnings('ignore')  # ignore warnings from sklearn

        # ------ update log file ----
        log_file = open("experiment_no_{}_{}_details.txt".format(
            exp_num, trial_num), "a")  # open log file
        GP_model_details = "\n" + "\n" + \
                           "          Calculating GP model . . . . .\n" "\n" + "\n"
        log_file.write(GP_model_details)
        log_file.close()  # close log file

        # ------------------------------------------------
        #
        #               Kernel for GP
        #
        # ------------------------------------------------
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))

        gp = GPR(kernel=kernel, alpha=.1, n_restarts_optimizer=10,
                 normalize_y=True)
        gp.fit(X_step, R_step)

        # --------- Fit GP model ----------------------------
        # predict UCB from gaussian process
        # create all possible combinations of state space variables
        X1_statespace, X2_statespace = X.flatten(), Y.flatten()
        X1_statespace, X2_statespace = X1_statespace.reshape(-1, 1), X2_statespace.reshape(-1, 1)
        X_statespace = np.concatenate((X1_statespace, X2_statespace), axis=1)
        # predict means and variances over whole state space to be analyzed by acquisition function
        R_GP_current, R_var_current = gp.predict(X_statespace, return_std=True)

        np.save(f'dump/GP_reward_model_mean_{iter_num}.npy', R_GP_current)
        np.save(f'dump/GP_reward_model_var_{iter_num}.npy', R_var_current)

        # ------ update log file ----
        log_file = open("experiment_no_{}_{}_details.txt".format(
            exp_num, trial_num), "a")  # open log file
        GP_model_details = "\n" + "\n" + \
                           "          Getting suggestion from acquisition function . . . . .\n" "\n" + "\n"
        log_file.write(GP_model_details)
        log_file.close()  # close log file

        # obtain suggestion from acquisition function
        suggestion = UCB(iter_num, R_GP_current, R_var_current,
                         X_statespace, iter_of_convergence)

        X_step = np.vstack((X_step, suggestion))  # append suggestion onto stack

        artifact_to_be_printed = suggestion

        np.save('data/suggestions_history.npy', X_step)

    np.save('data/suggestions_history.npy', X_step)

    state = X_step[-1]
    artifact_dimension = artifact_to_be_printed

    # ------- update log file ------------------
    # open log file
    log_file = open("experiment_no_{}_{}_details.txt".format(exp_num, trial_num), "a")
    # update the log file
    current_artifact_details = "##### epoch: {} -----------------------------  \n" \
                               "      BO feedback: ------------> \n" \
                               "              current state: {} \n" \
                               "              artifact to be printed now: {} \n".format(
                                iter_num, state, artifact_dimension)
    log_file.write(current_artifact_details)
    log_file.close()
    # ------------------------------
    artifact_to_be_printed = artifact_to_be_printed.tolist()
    if iter_num == 0:
        artifact_to_be_printed = artifact_to_be_printed[0]

    return artifact_to_be_printed


def run_update(iter_num, trial_num):
    exp_num = 5  # DO NOT CHANGE, this is fixed!!!
    # obtain reward value from AMSPnC and store in reward history stack
    # to help with indexing
    lxy = np.arange(700, 1100, 50).tolist()
    dia = np.arange(350, 650, 50).tolist()

    # load in suggestion data
    X_step = np.load('data/suggestions_history.npy')
    suggestion = X_step[-1]
    lxy_index = np.where(lxy == suggestion[1])
    dia_index = np.where(dia == suggestion[0])

    # load in reward associated with the latest suggestion
    # Generate data to substitute reading reward.csv file
    #X, Y, Rs, lxy, dia = gen_data()
    # find reward associated with suggestion
    #suggestion_reward = Rs[lxy_index, dia_index] + np.random.normal(loc=0, scale=.2)

    #if iter_num == 0:
    #    R_step = suggestion_reward
    #else:
    #    R_step = np.load('data/BO_reward_history.npy')
    #    R_step = np.vstack((R_step, suggestion_reward))

    #np.save('data/BO_reward_history.npy', R_step)

    # When integrating into machine, obtain reward data from AMSPnC ------------------
    reward_history = []
    with open('data/reward_history.csv', newline='') as csv_file: #uncomment section when not simulating
        reward_reader = csv.reader(csv_file, delimiter=',')
        for row in reward_reader:
            reward_history.append(float(row[0]))

    temp = np.array(reward_history)
    Y_step = temp.reshape(-1,1)
    np.save('data/BO_reward_history.npy', Y_step)
    # ---------------------------------------------------------------------------------

    return None

