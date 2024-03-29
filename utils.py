from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import os


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def plot(datas, gmm, fig_name="default.png", plot_dir="plots"):
    plt.clf()
    for data in datas:
        sns.distplot(data, kde=True, hist=True, bins=100, hist_kws={"alpha": 0.5})
    if gmm:
        max = np.max(np.concatenate(datas))
        min = np.min(np.concatenate(datas))
        x_axis = np.arange(min, max, 0.001)
        for mu, sigma in zip(gmm.mu, gmm.sigma):
            plt.plot(x_axis, norm.pdf(x_axis, mu, sigma), color="black")
    plt.title(fig_name.split(".")[0])
    plt.savefig(os.path.join(plot_dir, fig_name))
