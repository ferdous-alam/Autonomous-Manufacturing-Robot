from algorithm import algo
from Test_policy import test_policy

# ----- author details ------------------
__author__ = "Md Ferdous Alam, HRL, MAE, OSU"
__copyright__ = "Copyright 2021, MADE materials project, funded by NSF"
__version__ = "1.0"
__maintainer__ = "Md Ferdous Alam"
__email__ = "alam.92@osu.edu"
# ---------------------------------------


def run_algo(iter_num):

    # Experiment number: 04
    # Experiment name:
    # PRM-TAPRL algorithm

    # run Brain_PRM-TAPRL algorithm
    indices = algo.run_PRM_TAPRL_feedback(iter_num, trial_num=3)  # when training the agent
    # indices = test_policy.run_learned_policy_feedback(iter_num, trial_num=1)  # when testing the learned policy

    return indices


def run_update(iter_num):

    # Experiment number: 04
    # Experiment name:
    # PRM-TAPRL algorithm

    # run TAPRL algorithm
    algo.run_PRM_TAPRL_update(iter_num, trial_num=3)    # when training the agent
    # test_policy.run_learned_policy_update(iter_num, trial_num=1)  # when testing the learned policy

    return None


if __name__ == "__main__":
    import numpy as np
    import csv
    from lib.indices_to_artifact_dims import artifact_to_indices
    from lib.get_reward_from_AMSPnC_data import dummy_reward
    for i in range(24):
        val = run_algo(i)
        indices = artifact_to_indices(val)
        reward = dummy_reward(indices)
        print(f'sample: {val}, reward: {reward}')
        run_update(i)
    print('finished!')
