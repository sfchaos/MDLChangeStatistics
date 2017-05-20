from abc import ABCMeta, abstractmethod
import numpy as np

def heaviside(x):
    return 0.5 * (np.sign(x) + 1)

def slope(x):
    if x < 0:
        return 0.0
    elif 0 <= x <= 300:
        return x/300
    else:
        return 1.0


class DataGenerator():
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate(self, T, param, seed=None):
        pass

class SingleJumpingMeanGenerator(DataGenerator):
    def generate(self, T, delta_mu, seed=None):
        np.random.seed(seed)
        tau = int(T/2)
        mu = np.array(tau*[0.0] + tau*[delta_mu])
        x = np.array([np.random.normal(mu_i, 1.0) for mu_i in mu])
        return x

class SingleJumpingVarianceGenerator(DataGenerator):
    def generate(self, T, log_delta, seed=None):
        np.random.seed(seed)
        tau = int(T/2)
        log_sigma = np.array(tau*[0.0] + tau*[log_delta])
        sigma = np.exp(log_sigma)
        x = np.array([np.random.normal(0.0, sigma_i) for sigma_i in sigma])
        return x

class SingleJumpingMeanVarianceGenerator(DataGenerator):
    def generate(self, T, delta_mu, log_delta, seed=None):
        np.random.seed(seed)
        tau = int(T/2)
        mu = np.array(tau*[0.0] + tau*[delta_mu])
        #log_sigma = np.array(tau*[0.0] + tau*[log_delta])
        log_sigma = np.array(int(1.5*tau)*[0.0] + int(0.5*tau)*[log_delta])
        sigma = np.exp(log_sigma)
        x = np.array([np.random.normal(mu_i, sigma_i) for mu_i, sigma_i in zip(mu, sigma)])
        return x

class MultipleJumpingMeanGenerator(DataGenerator):
    def generate(self, T, coef=0.6, seed=None):
        np.random.seed(seed)
        mu = np.array([coef * np.sum(np.arange(9, 0, -1) * np.array([heaviside(n-1000*i) for i in range(9)])) \
                      for n in range(T)])
        x = np.array([np.random.normal(mu_i, 1) for mu_i in mu])
        return x

class MultipleJumpingVarianceGenerator(DataGenerator):
    def generate(self, T, coef=0.3, seed=None):
        np.random.seed(seed)
        log_sigma = np.array([coef * np.sum(np.arange(9, 0, -1) * np.array([heaviside(n-1000*i) for i in range(9)])) \
                              for n in range(T)])
        sigma = np.exp(log_sigma)
        x = np.array([np.random.normal(0, sigma_i) for sigma_i in sigma])
        return x

class MultipleGradualMeanGenerator(DataGenerator):
    def generate(self, T, coef=0.6, seed=None):
        np.random.seed(seed)
        mu = np.array([coef * np.sum(np.arange(9, 0, -1) * np.array([slope(n-1000*i) for i in range(9)])) \
                       for n in range(T)])
        x = np.array([np.random.normal(mu_i, 1) for mu_i in mu])
        return x

class MyMultipleGradualMeanGenerator(DataGenerator):
    def generate(self, T, coef=0.6, seed=None):
        np.random.seed(seed)
        mu = np.array([coef * np.sum(np.arange(9, 0, -1) * np.array([slope(n-x) for x in [1000, 1500, 2500, 3000, 3300, 3500, 4000, 5000, 9000]])) \
                      for n in range(T)])
        x = np.array([np.random.normal(mu_i, 1) for mu_i in mu])
        return x

class MultipleGradualVarianceGenerator(DataGenerator):
    def generate(self, T, coef=0.6, seed=None):
        np.random.seed(seed)
        log_sigma = np.array([coef * np.sum(np.arange(9, 0, -1) * np.array([slope(n-1000*i) for i in range(9)])) \
                              for n in range(T)])
        sigma = np.exp(log_sigma)
        x = np.array([np.random.normal(0, sigma_i) for sigma_i in sigma])
        return x
