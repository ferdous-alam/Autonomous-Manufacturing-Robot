import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
import warnings,time

from algorithm.acquisition import UCB
from visualizations import visualize_samples as vo

from data.gen_data import gen_data


def run_feedback(iter_num):

    exp_num = 2  # DO NOT CHANGE, this is fixed!!!

    #to help with indexing
    lxy = np.arange(700,1100,50)
    dia = np.arange(350,650,50)
    lxy = lxy.tolist()
    dia = dia.tolist()



    if iter_num == 0: #return initial condition to be printed as first iteration

        all_initial_states = []
        # get the fixed initial condition for the first iteration
        # initial_state = [1, 2]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        # initial_state = [1, 6]  # s = [d, lxy] ---> DO NOT CHANGE!!! This is fixed!!
        initial_state = [3, 1]

        dia_val = dia[initial_state[0]] # find true values corresponding to IC indices
        lxy_val = lxy[initial_state[1]]

        X_step = np.array([[dia_val, lxy_val]])
        artifact_to_be_printed = X_step




    else: #load previously sampled data and associated rewards, initialize gaussian process and call acquisition function to find maximum UCB value

        R_step = np.load('data/BO_reward_history.npy') #remove when not simulating

        #obtain current sample data
        X_step = np.load('data/suggestions_history.npy')

        # Initialize gaussian process
        warnings.filterwarnings('ignore') #ignore warnings from sklearn

        kernel = 1 * RBF(1, (1e-2, 1e2))
        gp = GPR(kernel=kernel, alpha=.1, n_restarts_optimizer=10, normalize_y=False)
        gp.fit(X_step, R_step)
        iter_of_convergence = 60; # set iteration at which convergence is desired

        #predict UCB from gaussian process
        lxy = np.arange(700, 1100, 50) #obtain state space values where function will be sampled
        dia = np.arange(350, 650, 50)
        X, Y = np.meshgrid(dia, lxy)

        X1_statespace, X2_statespace = X.flatten(), Y.flatten() #create all possible combinations of state space variables
        X1_statespace, X2_statespace = X1_statespace.reshape(-1, 1), X2_statespace.reshape(-1, 1)
        X_statespace = np.concatenate((X1_statespace, X2_statespace), axis=1)
        R_GP_current, R_var_current = gp.predict(X_statespace, return_std=True) #predict means and variances over whole state space to be analyzed by acquisition function

        suggestion = UCB(iter_num,R_GP_current,R_var_current,X_statespace,iter_of_convergence) #obtain suggestion from acquisition function

        X_step = np.vstack((X_step, suggestion)) #append suggestion onto stack


        artifact_to_be_printed = suggestion

        np.save('data/suggestions_history.npy', X_step)



    np.save('data/suggestions_history.npy', X_step)


    return artifact_to_be_printed

def run_update(iter_num): #obtain reward value from AMSPnC and store in reward history stack
    #to help with indexing
    lxy = np.arange(700,1100,50)
    dia = np.arange(350,650,50)
    lxy = lxy.tolist()
    dia = dia.tolist()


    #load in suggestion data
    X_step = np.load('data/suggestions_history.npy')
    suggestion = X_step[-1]
    lxy_index = np.where(lxy == suggestion[1])
    dia_index = np.where(dia == suggestion[0])

    #load in reward associated with latest suggestion
    #Generate data to substitute reading reward.csv file
    X, Y, Rs, lxy, dia = gen_data()
    suggestion_reward = Rs[lxy_index,dia_index] + np.random.normal(loc=0,
                                                                    scale=.2)  # find reward associated with suggestion

    if iter_num == 0:
        R_step = suggestion_reward

    else:
        R_step = np.load('data/BO_reward_history.npy')
        R_step = np.vstack((R_step, suggestion_reward))

    np.save('data/BO_reward_history.npy', R_step)


    #---------------------When integrating into machine, obtain reward data from AMSPnC --------------------------------------------

    # reward_history = []
    # with open('data/reward_history.csv', newline='') as csv_file: #uncomment section when not simulating
    # reward_reader = csv.reader(csv_file, delimiter=',')
    # for row in reward_reader:
    # reward_history.append(float(row[0]))

    # temp = np.array(reward_history[0:iter_num:1]).reshape(-1,1)
    # Y_step = temp;
    # np.save('data/BO_reward_history.npy', Y_step)
    #--------------------------------------------------------------------------------- --------------------------------------------

    return None

