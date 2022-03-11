from algorithm.algo import run_feedback,run_update
import numpy as np
from visualizations import visualize_samples as vo
from visualizations.visualize_GP import plotGP




if __name__ == "__main__":
    lxy = np.arange(700,1100,50)
    dia = np.arange(350,650,50)
    #to simulate AMSPnC, run in a loop.

    for ii in range(0,60):

        #AMSPnC asks algorithm for sample to print
        artifact_to_be_printed = run_feedback(ii)
        print("print: "+ str(artifact_to_be_printed)+" on iter: "+str(ii+1))



        #AMSPnC prints sample, tests, and writes reward to csv file
        run_update(ii)

        if ii%10 == 0:
            plotGP(ii)





    states = np.load('data/suggestions_history.npy')
    plot = []
    for i in range(0,len(states)):
        dia_idx = np.where(dia == states[i,0])[0]
        lxy_idx = np.where(lxy== states[i,1])[0]

        plot.append([dia_idx[0],lxy_idx[0]])


    vo.visualize_samples(i,plot)


