import numpy as np
from numpy.lib.stride_tricks import as_strided

from smdl import SMDL
from model import Norm1D, Poisson1D
from data_generator import SingleJumpingMeanGenerator, SingleJumpingVarianceGenerator, \
                           SingleJumpingMeanVarianceGenerator, \
                           MultipleJumpingMeanGenerator, MultipleJumpingVarianceGenerator, \
                           MultipleGradualMeanGenerator, MultipleGradualVarianceGenerator
from evaluator import BenefitFalseAlarmEvaluator
#import changefinder

from sklearn.metrics import auc
import matplotlib.pyplot as plt

import sys

def calc_score(x, h=100, T=1000, mu_max=2.0, sigma_min=1.0, beta=1.0):
    # stride data
    x_strided = as_strided(x, (T-2*h, 2*h), (8, 8))
    # model
    model = Norm1D(h, T)
    # sequential MDL-change detection algorithm
    smdl = SMDL(h, T, model, beta)
    score = []
    for i in range(x_strided.shape[0]):
        score_i = smdl.calc_change_score(x_strided[i, :], mu_max, sigma_min)
        score.append(score_i)

    return score


def calc_auc(score, h=100, tol_delay=200, t_change=[500]):
    t_change = np.array(t_change)
    evaluator = BenefitFalseAlarmEvaluator()
    total_benefit = []
    num_false_alarm = []
    for beta in np.linspace(np.min(score), np.max(score), 100):
        tb, nfa = evaluator.evaluate(score, tol_delay, t_change, h, beta)
        total_benefit.append(tb)
        num_false_alarm.append(nfa)

    total_benefit_npa = np.array(total_benefit)
    num_false_alarm_npa = np.array(num_false_alarm)

    tpr = total_benefit_npa / np.max(total_benefit_npa)
    if not np.all(num_false_alarm_npa == 0.0):
        fpr = num_false_alarm_npa / np.max(num_false_alarm_npa)
    else:
        tpr = np.hstack((1.0, tpr))
        fpr = np.hstack((1.0, num_false_alarm_npa))

    print(auc(fpr[::-1], tpr[::-1]))


def test_single(h=100, T=1000, beta=0.5):
    # generate data
    """
    # single mean-changing dataset
    gen_single_mean = SingleJumpingMeanGenerator()
    delta_mu = 0.5
    x = gen_single_mean.generate(T, delta_mu)
    mu_max = 2.0
    sigma_min = 1.0
    """

    """
    # single variance-changing dataset
    gen_single_var = SingleJumpingVarianceGenerator()
    log_delta = 1.0
    x = gen_single_var.generate(T, log_delta)
    mu_max = 1.0
    sigma_min = np.exp(0.25)
    """

    # single mean-variance-changing dataset
    gen_single_mean_var = SingleJumpingMeanVarianceGenerator()
    delta_mu = 0.5
    log_delta = 0.25
    x = gen_single_mean_var.generate(T, delta_mu, log_delta)
    mu_max = 2.0
    sigma_min = np.exp(0.25)

    tol_delay = 200
    t_change = 500
    score = calc_score(x, h, T, mu_max, sigma_min)
    calc_auc(score, h, tol_delay, t_change)


def test_multiple(h=100, T=10000, beta=0.5):
    # generate data
    t_change = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

    # multiple jumping mean-changing dataset
    gen_multi_mean = MultipleJumpingMeanGenerator()
    x = gen_multi_mean.generate(T)
    mu_max = 2.0
    sigma_min = 1.0

    """
    # multiple jumping variance-changing dataset
    gen_multi_var = MultipleJumpingVarianceGenerator()
    x = gen_multi_var.generate(T)
    mu_max = 1.0
    sigma_min = np.exp(0.25)
    """

    """
    # multiple gradual mean-changing dataset
    gen_multi_mean = MultipleGradualMeanGenerator()
    x = gen_multi_mean.generate(T)
    mu_max = 2.0
    sigma_min = 1.0
    """

    """
    # multiple gradual variance-changing dataset
    gen_multi_var = MultipleGradualVarianceGenerator()
    x = gen_multi_var.generate(T)
    mu_max = 1.0
    sigma_min = np.exp(0.25)
    """

    tol_delay = 200
    score = calc_score(x, h, T, mu_max, sigma_min)
    calc_auc(score, h, tol_delay, t_change)


if __name__ == '__main__':
    test_single(h=200)
    #test_multiple()
