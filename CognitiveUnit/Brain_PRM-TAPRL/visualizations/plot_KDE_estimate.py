import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def plot_kde_estimate(rewards_cache, rewards_pred, rewards_kde_mean, rewards_kde_std,
                      iter_num=None, save_plot=False):
    rows = (len(rewards_kde_mean) // 2) + 1

    f = plt.figure(figsize=(10, 16))

    for k in range(len(rewards_kde_mean)):
        y = np.array(rewards_cache[k])
        x = np.linspace(min(y) - 1.0, max(y) + 1.0, 100)
        y_n = stats.norm.pdf(x, rewards_kde_mean[k], rewards_kde_std[k])
        y_n = y_n / sum(y_n)

        # figure properties
        ax = f.add_subplot(rows, 2, k + 1)
        plt.rcParams['text.usetex'] = True
        plt.fill_between(x, rewards_pred[k], alpha=0.5)
        plt.plot(y, np.full_like(y, -0.0025), 'ok', markersize=3.5, alpha=0.5)
        plt.fill_between(x, y_n, alpha=0.5)

        # additional properties
        plt.xlabel(r'$\mathcal{R}(\mathbf{x})$', fontsize=14)
        plt.rcParams['axes.linewidth'] = 1.25
        plt.xticks(fontsize=14)
        plt.yticks([], fontsize=14)

    if save_plot:
        # save plot as pdf
        if iter_num:
            plt.savefig('figures/plot_kde_estimate_{}.pdf'.format(iter_num))
