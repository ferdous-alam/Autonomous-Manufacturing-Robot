from algorithm import algo

# ----- author details ------------------
__author__ = "Md Ferdous Alam, HRL, MAE, OSU"
__copyright__ = "Copyright 2021, MADE materials project, funded by NSF"
__version__ = "1.0"
__maintainer__ = "Md Ferdous Alam"
__email__ = "alam.92@osu.edu"
# ---------------------------------------


def run_algo(iter_num):

    # Experiment number: 02
    # Experiment name:
    # policy from vanilla Q-learning (Brain_Q-learning)

    # run Brain_PRM-TAPRL algorithm
    indices = algo.run_Q_learning_feedback(iter_num)

    return indices


def run_update(iter_num):

    # Experiment number: 01
    # Experiment name:
    # Offline execution of source optimal policy (Brain_PRM-TAPRL)

    # run Brain_PRM-TAPRL algorithm
    algo.run_Q_learning_update(iter_num)

    return None


# if __name__ == "__main__":
#    for i in range(3):
#        val = run_algo(i)
#        print(val)
#        run_update(i)

