from algorithm import PRM_TAPRL

# ----- author details ------------------
__author__ = "Md Ferdous Alam, HRL, MAE, OSU"
__copyright__ = "Copyright 2021, MADE materials funded by NSF"
__version__ = "1.0"
__maintainer__ = "Md Ferdous Alam"
__email__ = "alam.92@osu.edu"
# ---------------------------------------


def run_algo(iter_num):
    # Experiment number: 01
    # Experiment name:
    # Probabilistic reward modeling for temporal abstractions in reinforcement learning (PRM-TAPRL)

    # run PRM-TAPRL algorithm
    num_of_options, option_epsilon, H = 5, 0.75, 5
    algo_params = [num_of_options, option_epsilon, H]
    iter_num = 0
    artifact_indices = PRM_TAPRL.run_initial_PRM_TAPRL(iter_num, algo_params)

    return artifact_indices
