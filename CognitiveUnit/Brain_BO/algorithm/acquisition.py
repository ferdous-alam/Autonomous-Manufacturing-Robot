import numpy as np


def UCB(iter_num, R_GP_current, R_var_current,
        X_statespace, iter_of_convergence):
    """
    UCB acquisition function for Bayesian optimization
    """

    # beta = 2*(iter_num * 1 / iter_of_convergence)
    beta = iter_of_convergence / (iter_num + 1)
    print(f'beta:{beta}, iter_num:{iter_num}')

    maximization_variable = np.add(R_GP_current, beta * R_var_current.reshape(-1, 1))
    UCB_max = np.max(maximization_variable)

    max_index = np.where(maximization_variable == UCB_max)
    max_index = max_index[0]

    suggestion = X_statespace[max_index]  # find state to suggest
    num_suggestions = suggestion.shape[0]

    suggestion = suggestion[
        np.random.randint(0, num_suggestions)]  # if more than one value correspond to maximum, random sample

    return suggestion
