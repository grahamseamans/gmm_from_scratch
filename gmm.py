import numpy as np
from scipy.stats import norm
from utils import softmax
import functools
from tqdm import tqdm


class GMM:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mu = np.random.rand(self.n_components)
        self.sigma = np.ones(self.n_components)
        self.pi = softmax(np.random.rand(self.n_components))

    def fit(self, X, n_iter=20):
        for _ in tqdm(range(n_iter)):
            r = self._e(X)
            self._m(X, r)

    @functools.cache
    def expectation(self, p, x, m, s):
        return p * norm.pdf(x, m, s)

    # for each data point find which component is closest. (we're going to make it closer)
    def _e(self, X):
        n_samples = len(X)
        r = np.zeros((n_samples, self.n_components))
        for i in range(n_samples):  # an unvectorized horror
            for j in range(self.n_components):
                r[i, j] = self.expectation(
                    self.pi[j], X[i], self.mu[j], self.sigma[j]
                ) / sum(
                    [
                        self.expectation(self.pi[k], X[i], self.mu[k], self.sigma[k])
                        for k in range(self.n_components)
                    ]
                )

        return r

    def _m(self, X, r):
        n_samples = len(X)
        for j in range(self.n_components):
            self.mu[j] = sum(r[:, j] * X) / sum(r[:, j])
            self.sigma[j] = np.sqrt(sum(r[:, j] * (X - self.mu[j]) ** 2) / sum(r[:, j]))
            self.pi[j] = sum(r[:, j]) / n_samples
