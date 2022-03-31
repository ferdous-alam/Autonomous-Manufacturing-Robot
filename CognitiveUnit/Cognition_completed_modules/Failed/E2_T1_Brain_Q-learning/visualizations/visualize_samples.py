import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_samples(iter_num, states):
    """
    Visualizes the last state from each option

    options states has the following form:
        data structure: hash table
        options = {[option 1 states], [option 2 states], .... [option m states]}
    """
    # plot properties -----------------------
    # plt.rc('text', usetex=True)  # use latex for all fonts in the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    # ---------------------------------------

    # extract option states
    lxy = np.arange(700, 1100, 50)
    dia = np.arange(350, 650, 50)
    states_x_idx = [states[i][0] for i in range(len(states))]
    states_y_idx = [states[i][1] for i in range(len(states))]
    states_x = [dia[states_x_idx[k]] for k in range(len(states_x_idx))]
    states_y = [lxy[states_y_idx[k]] for k in range(len(states_y_idx))]

    plt.scatter(states_y, states_x, 500, marker='o', facecolor='gray', edgecolor='blue', alpha=0.5)

    # plot properties
    plt.xlabel(r'$l_{xy} \ \ (\mu m)$', fontsize=25)
    plt.ylabel(r'$d \ \ (\mu m)$', fontsize=25)
    plt.xlim(650, 1100)
    plt.ylim(300, 650)
    plt.grid('on', linestyle='--', lw=2.0, alpha=0.20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)
    # plt.legend(fontsize=18)

    # save plot as pdf
    plt.savefig('figures/explored_states_iter_{}.pdf'.format(iter_num),
                format='pdf', bbox_inches='tight', dpi=1200)


# if __name__ == "__main__":
#     s = [[1, 2], [3, 3], [2, 4]]
#     visualize_samples(3, s)
