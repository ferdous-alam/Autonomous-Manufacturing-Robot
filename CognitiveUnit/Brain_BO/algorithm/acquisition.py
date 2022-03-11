import numpy as np

# -------------------acquistion function------------------------------------

def UCB(iter_num,R_GP_current,R_var_current,X_statespace,iter_of_convergence):


    beta = 4 - 2*(iter_num * 1 / iter_of_convergence)
    maximization_variable = np.add(R_GP_current, beta * R_var_current.reshape(-1, 1))
    UCBmax = np.max(maximization_variable)

    maxindex = np.where(maximization_variable == UCBmax)
    maxindex = maxindex[0]

    suggestion = X_statespace[maxindex]  # find state to suggest
    numsuggestions = suggestion.shape[0]

    suggestion = suggestion[
        np.random.randint(0, numsuggestions)]  # if more than one value correspond to maximum, random sample



    return suggestion