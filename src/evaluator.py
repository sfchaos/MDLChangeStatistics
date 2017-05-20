import numpy as np
from abc import ABCMeta, abstractmethod

class Evaluator:
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, x, T, t_changes, h, beta):
        pass


class BenefitFalseAlarmEvaluator(Evaluator):
    def evaluate(self, score, T, t_change, h, beta):
        """
        evaluate benefit of alarm and false alarm
        :param score: scores of a sequence
        :param T: maximum tolerant delay
        :param t_change: change point ¥
        :param h:
        :param beta:
        :return:
        """
        # binary alarms
        binary_alarm = (np.array(score) > beta)
        # benefit of an alarm at time t
        #benefit = np.array([np.max([1 - np.abs(t + h - t_change)/T, 0.0]) for t in range(len(score))])
        benefit = np.array([np.max(np.hstack((1 - np.abs(t + h - t_change)/T, 0.0))) for t in range(len(score))])
        # total benefit of alarm sequence
        total_benefit = np.sum(binary_alarm * benefit)
        # number of false alarms
        num_false_alarm = np.sum(binary_alarm * (benefit == 0.0))

        return total_benefit, num_false_alarm
