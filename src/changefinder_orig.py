import numpy as np

class YuleWalker:
    def __init__(self):
        pass

    def solve(self, r, lpcOrder):
        # Levinson-Durbin
        


class ChangeFinder:
    def __init__(self, dim, r, k, seed=None):
        """
        :param dim: dimension of variables
        :param r: discounting parameter
        :param k: order of AR model
        :param seed: random seed
        :return:
        """
        np.random.seed(seed)
        self.mu = [np.random.randn(dim) for _ in range(k)]
        self.C = [np.random.randn(dim, dim) for _ in range(k)]
        self.omega = [np.random.randn(dim) for _ in range(k)]
        self.r = r
        self.k = k
        self.sequence = []
        self.yw = YuleWalker()

    def update(self, x):
        #self.sequence.append(x)
        r = self.r
        k = self.k
        for j in range(k):
            # update mu, C
            self.mu[j] = (1 - r) * self.mu[j] + r * x
            self.C[j] = (1 - r) * self.C[j] + r * np.outer(x - self.mu, self.sequence[-j-1] - self.mu)

        # solve Yule-Walker equation
        self.omega = self.yw.solve()

        # update

