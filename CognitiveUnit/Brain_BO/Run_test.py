from algorithm.algo import run_feedback, run_update
import numpy as np
from visualizations import visualize_samples as vo
from visualizations.visualize_GP import plotGP


# This script runs a simulated test of the written BO algorithm
# Data is generated from the offline dataset of PnC artifacts

lxy = np.arange(700, 1100, 50)
dia = np.arange(350, 650, 50)

# to simulate AMSPnC, run in a loop.
T = 24
for k in range(T):
    # AMSPnC asks algorithm for sample to print
    artifact_to_be_printed = run_feedback(k, trial_num=2,
                                          iter_of_convergence=T)
    print("print: " + str(artifact_to_be_printed)+" on iter: "+str(k+1))

    # AMSPnC prints sample, tests, and writes reward to csv file
    run_update(k, trial_num=2)

    # only plot GP model at every 10 iterations
    if k % 10 == 0:
        plotGP(k)

        states = np.load('data/suggestions_history.npy')
        plot = []
        for i in range(len(states)):
            dia_idx = np.where(dia == states[i, 0])[0]
            lxy_idx = np.where(lxy == states[i, 1])[0]

            plot.append([dia_idx[0], lxy_idx[0]])

        vo.visualize_samples(k, plot)


