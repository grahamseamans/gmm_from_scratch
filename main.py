import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from utils import plot
from gmm import GMM

if __name__ == "__main__":
    num_dists = 3
    means = np.random.rand(num_dists) * 10 - 5
    vars = np.random.rand(num_dists) * 3 + 0.25
    datas = [np.random.normal(mean, var, 1000) for mean, var in zip(means, vars)]
    population = np.concatenate(datas)
    plot(datas, None, "dists.png")
    GMM = GMM(num_dists)
    plot(datas, GMM, "pre_fit.png")
    GMM.fit(population)
    plot(datas, GMM, "post_fit.png")
