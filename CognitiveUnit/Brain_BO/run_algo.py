from algorithm import algo
from visualizations import visualize_samples as vo
from visualizations.visualize_GP import plotGP
import numpy as np
# ----- author details ------------------
__author__ = "Md Ferdous Alam, Sarp Sezer, HRL, MAE, OSU"
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
    indices = algo.run_feedback(iter_num, trial_num=1)

    return indices


def run_update(iter_num):

    # Experiment number: 01
    # Experiment name:
    # Offline execution of source optimal policy (Brain_PRM-TAPRL)

    # run Brain_PRM-TAPRL algorithm
    algo.run_update(iter_num, trial_num=1)

 
    
    plotGP(iter_num)
    lxy = np.arange(700, 1100, 50)
    dia = np.arange(350, 650, 50)
    states = np.load('data/suggestions_history.npy')
    plot = []
    for i in range(len(states)):
        dia_idx = np.where(dia == states[i, 0])[0]
        lxy_idx = np.where(lxy == states[i, 1])[0]

        plot.append([dia_idx[0], lxy_idx[0]])

    vo.visualize_samples(iter_num, plot)

    return None


