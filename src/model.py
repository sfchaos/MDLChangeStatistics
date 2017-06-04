import numpy as np
from scipy.special import gamma, gammaln

def gammaln_m(x, m):
    """
    \gamma_{m}(x) = pi^(m(m-1)/4) * \prod_{j=1}^{m} \gamma(x + (1-j)/2)
    \log{\gamma_{m}(x)} = m(m-1)/4 * \log{\pi} + \sum_{j=1}^{m} \log{\gamma(x+(1-j)/2)}
    :param x:
    :param m:
    :return:
    """
    return m*(m-1)/4 * np.log(np.pi) + np.sum([gammaln(x + (1-j)/2) for j in range(1, m+1)])

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


class Norm(Model):
    def __init__(self, h, T):
        """
        initialize parameters
        :param h: window size
        :param T: sequence length
        :return:
        """
        super().__init__(h, T)

    def _log_NML_normalizer(self, k, m, mu_max, sigma_min):
        """
        Normalized Maximum Likelihood normalizer
        :param k:
        :param m:
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: log normalized maximum likelihood normalizer
        """
        return -(m+1)*np.log(m/2) + m/2*np.log(mu_max) - m/2*np.sum(np.log(sigma_min)) + \
               m*k/2.0*(np.log(k/2) - 1.0) - gammaln(m/2) - gammaln_m((k-1)/2, m)
               #m*k/2.0*(np.log(k/2) - 1.0) - gammaln(m/2) - gammaln((k-1)/2)

    def calc_change_score(self, x, mu_max, sigma_min):
        """
        calculate change score for a given point
        :param x: point (time points * features)
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: change statistic
        """
        h = self.h

        mu0 = np.mean(x, axis=0)
        mu1 = np.mean(x[:h, :], axis=0)
        mu2 = np.mean(x[h:, :], axis=0)

        cor0 = np.corrcoef(x, rowvar=0)
        cor1 = np.corrcoef(x[:h, :], rowvar=0)
        cor2 = np.corrcoef(x[h:, :], rowvar=0)

        cor_inv0 = np.linalg.inv(cor0)
        cor_inv1 = np.linalg.inv(cor1)
        cor_inv2 = np.linalg.inv(cor2)

        dim = x.shape[1]

        # calculate change statistic
        term0 = dim * h * np.log(2.0 * np.pi) + h * np.linalg.det(cor0) + \
                np.sum([0.5 * (xx - mu0).reshape(1, -1).dot(cor_inv0).dot(xx - mu0) for xx in x])
        term1 = dim * h/2 * np.log(2.0 * np.pi) + 0.5 * h * np.linalg.det(cor1) + \
                np.sum([0.5 * (xx - mu1).reshape(1, -1).dot(cor_inv1).dot(xx - mu1) for xx in x[:h, :]])
        term2 = dim * h/2 * np.log(2.0 * np.pi) + 0.5 * h * np.linalg.det(cor2) + \
                np.sum([0.5 * (xx - mu2).reshape(1, -1).dot(cor_inv2).dot(xx - mu2) for xx in x[h:, :]])
        change_statistic = term0 - term1 - term2 + \
                           self._log_NML_normalizer(2*h, dim, mu_max, sigma_min) - \
                           2.0 * self._log_NML_normalizer(h, dim, mu_max, sigma_min)
        return change_statistic


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

class LinearRegression(Model):
    def __init__(self, h, T):
        super().__init__(h, T)

    def calc_change_score(self, x, sigma_min=1.0, R=10.0):
        h = self.h
        sigma0 = np.std(x)
        sigma1 = np.std(x[:h])
        sigma2 = np.std(x[h:])

        # calculate change statistic
        change_statistic = h * np.log(sigma0**2/(sigma1 * sigma2)) - \
                            np.log(R/sigma_min**2) - \
                            np.log(gammaln(h-1)) - 2.0 * np.log(gammaln(h/2-1)) + \
                            h * np.log(2.0)
        return change_statistic
