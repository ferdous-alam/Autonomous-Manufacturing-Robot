import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_option_states(states, iter_num):
    """
    Visualizes the last state from each option

    options states has the following form:
        data structure: hash table
        options = {[option 1 states], [option 2 states], .... [option m states]}
    """
    # plot properties -----------------------
    plt.rc('text', usetex=True)  # use latex for all fonts in the figure
    plt.figure(figsize=(10, 8))  # figure size
    # ---------------------------------------

    # extract option states
    lxy = np.arange(700, 1100, 50)
    dia = np.arange(350, 650, 50)
    for i in range(len(states)):
        states_x_idx = [states[i][j][0] for j in range(len(states[i]))]
        states_y_idx = [states[i][j][1] for j in range(len(states[i]))]
        states_x = [dia[states_x_idx[k]] for k in range(len(states_x_idx))]
        states_y = [lxy[states_y_idx[k]] for k in range(len(states_y_idx))]
        if i == 0:  # for showing the legend
            plt.scatter(states_y[-1], states_x[-1], 250, marker='s', color='gray',
                        alpha=0.5,  edgecolors='blue', label=r'artifacts')
        else:
            plt.scatter(states_y[-1], states_x[-1], 250, marker='s', color='gray',
                        alpha=0.5, edgecolors='blue')
    # plot properties
    plt.xlabel(r'$l_{xy} \ \ (\mu m)$', fontsize=25)
    plt.ylabel(r'$d \ \ (\mu m)$', fontsize=25)
    plt.xlim(650, 1100)
    plt.ylim(300, 650)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.rcParams['axes.linewidth'] = 3.0
    plt.legend(fontsize=18)

    # save plot as pdf
    plt.savefig('figures/option_states_iter_{}.pdf'.format(iter_num),
                format='pdf', bbox_inches='tight', dpi=1200)


if __name__ == "__main__":
    option_states = {0: [[1, 2], [2, 1], [3, 0], [4, 1], [4, 1]],
                     1: [[1, 2], [2, 3], [3, 3], [2, 4], [1, 4]],
                     2: [[1, 2], [2, 3], [2, 4], [2, 4], [2, 4]]}
    visualize_option_states(option_states, iter_num=1)
