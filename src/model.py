import numpy as np
from scipy.special import gamma, gammaln

class Model():
    def __init__(self, h, T):
        """
        initialize parameters
        :param h: window size
        :param T: sequence length
        :return:
        """
        self.h = h
        self.T = T


class Norm1D(Model):
    def __init__(self, h, T):
        """
        initialize parameters
        :param h: window size
        :param T: sequence length
        :return:
        """
        super().__init__(h, T)

    def _log_NML_normalizer(self, k, mu_max, sigma_min):
        """
        Normalized Maximum Likelihood normalizer
        :param k:
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: log normalized maximum likelihood normalizer
        """
        return 0.5 * np.log(16 * np.abs(mu_max) / (np.pi * sigma_min**2)) + \
               k/2 * np.log(k/2) - k/2 - gammaln((k-1)/2)
               #k/2 * np.log(k/2) - k/2 - np.log(gamma((k-1)/2))

    def calc_change_score(self, x, mu_max, sigma_min):
        """
        calculate change score for a given point
        :param x: point
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: change statistic
        """
        h = self.h
        sigma0 = np.std(x)
        sigma1 = np.std(x[:h])
        sigma2 = np.std(x[h:])

        # calculate change statistic
        change_statistic = h/2 * np.log(sigma0**2/(sigma1 * sigma2)) + \
                           self._log_NML_normalizer(2*h, mu_max, sigma_min) - \
                           2.0 * self._log_NML_normalizer(h, mu_max, sigma_min)
        return change_statistic


class Poisson1D(Model):
    def __init__(self, h, T):
        super().__init__(h, T)

    def _log_NML_normalizer(self, k, lambda_max):
        return 0.5 * np.log(k/(2*np.pi)) + (1 + lambda_max/2)*np.log(2) + np.log(lambda_max)

    def calc_change_score(self, x):
        """
        calculate change score for a given point
        :param x: point
        :return:
        """
        h = self.h
        lambda0 = np.mean(x)
        lambda1 = np.mean(x[:h])
        lambda2 = np.mean(x[h:])

        # calculate change statistic
        change_statistic = -2*h*(lambda0*np.log(lambda0) - 0.5*(lambda1*np.log(lambda1) + lambda2*np.log(lambda2))) + \
                           self._log_NML_normalizer(2*h) - self._log_NML_normalizer(h)
        return change_statistic
