import numpy as np
import sys

class YuleWalker:
    def __init__(self):
        pass

    def solve(self, C):
        k = len(C) - 1
        omega = []
        C0_inv = np.linalg.inv(C[0])
        # k = 1
        omega.append(C[1].dot(C0_inv))
        # k >= 2
        for j in range(1, k):
            right_side = C[j+1]
            for i in range(j):
                right_side -= omega[i].dot(C[j-i])
            omega.append(right_side.dot(C0_inv))

        return omega

class SDAR:
    def __init__(self, dim, r=0.05, k=3, seed=None):
        """
        :param dim: dimension of variables
        :param r: discount rate
        :param k: order of AR model
        :param seed: random seed
        :return:
        """
        np.random.seed(seed)
        self.mu = np.random.rand(dim)
        self.sigma = np.random.rand(dim, dim)
        #self.C = [np.random.rand(dim, dim) for _ in range(k+1)]
        self.C = [np.zeros((dim, dim)) for _ in range(k+1)]
        #self.omega = [np.random.rand(dim, dim) for _ in range(k)]
        self.omega = [np.zeros((dim, dim)) for _ in range(k)]
        self.dim = dim
        self.r = r
        self.k = k
        self.yw = YuleWalker()

    def update(self, x, sequence, message):
        """
        update parameters
        :param x: data (dimension = self.dim)
        :param sequence: history of data (past 'x's)
        :return: log_score
        """
        r = self.r
        k = self.k
        # update mu, C
        self.mu += -r * self.mu + r * x
        if message == 'second':
            print(self.mu)
        # j = 0
        self.C[0] += -r * self.C[0] + r * np.outer(x - self.mu, x - self.mu)
        # j = 1, ... , k-1
        for j in range(1, k+1):
            self.C[j] += -r * self.C[j] + r * np.outer(x - self.mu, sequence[-j] - self.mu)

        # solve Yule-Walker equation
        self.omega = self.yw.solve(self.C)

        # update
        x_hat = np.sum(np.array([self.omega[i].dot(sequence[-i-1] - self.mu) + self.mu \
                                 for i in range(k)]), axis=0)
        self.sigma += -r * self.sigma + r * np.outer(x - x_hat, x - x_hat)

        x_minus_x_hat = (x - x_hat).reshape(-1, 1)
        det = np.linalg.det(self.sigma)
        sigma_inv = np.linalg.inv(self.sigma)
        log_score = self.dim/2 * np.log(2.0 * np.pi) + \
                    0.5 * np.log(np.abs(det)) + \
                    0.5 * x_minus_x_hat.T.dot(sigma_inv).dot(x_minus_x_hat)
        return float(log_score)


class MyChangeFinder:
    def __init__(self, dim, r=0.05, k=3, smooth=5, seed=None):
        """
        :param dim: dimension of variables
        :param r: discounting parameter
        :param k: order of AR model
        ;param smooth: order of smoothing
        :param seed: random seed
        :return:
        """
        np.random.seed(seed)
        self.sdar_1st = SDAR(dim, r, k)
        self.sdar_2nd = SDAR(dim, r, k)
        self.dim = dim
        self.r = r
        self.k = k
        self.smooth_1st = smooth
        self.smooth_2nd = int(smooth/2)
        self.sequence_1st = []
        self.scores_1st = []
        self.scores_2nd = []
        self.moving_average_scores_1st = []
        self.moving_average_scores_2nd = []
        self.t = 0

    def _smoothing(self, scores, smooth):
        score = np.mean(scores[-smooth:])
        return score

    def update(self, x):
        # Step1: first stage learning (SDAR)
        if self.t >= self.k:
            score_1st = self.sdar_1st.update(x, self.sequence_1st, 'first')
        else:
            score_1st = np.nan
        self.sequence_1st.append(x)
        self.scores_1st.append(score_1st)

        # Step2: smoothing
        if self.t >= self.k + self.smooth_1st:
            smooth_score_1st = self._smoothing(self.scores_1st, self.smooth_1st)
        else:
            smooth_score_1st = np.nan
        self.moving_average_scores_1st.append(smooth_score_1st)

        # Step3: second stage learning and smoothing
        if self.t >= 2*self.k + self.smooth_1st:
            score_2nd = self.sdar_2nd.update(self.moving_average_scores_1st[-1], self.moving_average_scores_1st[:-1], 'second')
        else:
            score_2nd = np.nan
        self.scores_2nd.append(score_2nd)
        # smoothing
        if self.t >= 2*self.k + self.smooth_1st + self.smooth_2nd:
            smooth_score_2nd = self._smoothing(self.scores_2nd, self.smooth_2nd)
        else:
            smooth_score_2nd = np.nan
        self.moving_average_scores_2nd.append(smooth_score_2nd)

        self.t += 1
        return smooth_score_2nd
