"""
Created on Thu May 31 16:14:32 2018

@author: Siavash Hakim Elahi
"""
import numpy as np
import pandas as pd
import datetime
from scipy.stats import multivariate_normal

class AnomalyDetection():

    def __init__(self,tseries,events_real,ts_matrix):
        self.tseries = tseries
        self.events_real = events_real
        self.tsh = [.001]
        self.threshold = 0.05  # Relative change limit for the slope
        self.threshold = 0.02  # Relative change limit for the function value.
        self.sC = 0.001  # Stability term
        self.ts_matrix = ts_matrix


    def event_detection(self):
        """"
        This function finds the peaks that represent anomalies in timeseries.

        INPUT:  A list of numbers.
        OUTPUT: A list of 0's and 1's (numbers) where 1's indicate the presence of a jump
                and 0's indicate a lack of a jump.
        """
        tseries_list = self.tseries.tolist()
        n = len(tseries_list)
        jump_series = [0] * n
        # Calculate the slope. Use a forward difference. Set the last point's slope to 0.
        slope = [tseries_list[i + 1] - tseries_list[i] for i in range(0, n - 1)]
        slope.append(0)

        # Calculate the median.
        med = np.median(np.asarray(tseries_list)).tolist()

        # Set the maximum value to be a jump.
        max_indx = tseries_list.index(max(tseries_list))
        if tseries_list[max_indx] > 0:
            jump_series[max_indx] = 1

        # Look for relatively large positive changes in slope to indicate a jump.
        for i in range(1, n - 1):
            try:
                if abs((slope[i] - slope[i - 1]) / (slope[i - 1] + self.sC)) > self.threshold and slope[i] > 0 and (
                        tseries_list[i] - tseries_list[i - 1]) / (tseries_list[i - 1] + self.sC) > self.threshold2 and tseries_list[i] != tseries_list[
                    i - 1] and \
                        tseries_list[i] > med:
                    while tseries_list[i + 1] > tseries_list[i]:
                        i += 1
                    jump_series[i] = 1
            except:
                jump_series[i] = 0

        return jump_series


    def unsupervised_event_detection(self):
        """"
       This function finds the peaks that represent anomalies in multivariate timeseries.

       INPUT:  A matrix of time-series (ts_matrix). [series, series]
       OUTPUT: A list of 0's and 1's (numbers) where 1's indicate the presence of a jump
               and 0's indicate a lack of a jump.
       """
        mat_size = np.shape(self.ts_matrix)
        No_ts = mat_size[0]
        No_time = mat_size[1]

        muTT = []
        for j in range(0, No_ts):
            muTT.append(np.mean(self.ts_matrix[j]))

        varTT = []
        for j in range(0, No_ts):
            varTT.append(np.std(self.ts_matrix[j]))

        prob_eval = []
        for i in range(0, No_time):
            ts_stack = []
            for j in range(0, No_ts):
                ts_stack.append(self.ts_matrix[j][i])

            probx = multivariate_normal.pdf(ts_stack, mean=muTT, cov=varTT)
            prob_eval.append(probx)

        if len(self.tsh) == 1:
            Event_stoT = np.zeros(len(prob_eval))
            for i in range(0, len(prob_eval)):
                if prob_eval[i] < self.tsh[0]:
                    Event_stoT[i] = 1
        else:
            print("input the threshold value")

        EventstoT_index = [j for j, i in enumerate(Event_stoT) if Event_stoT[j] == 1]

        while len(EventstoT_index) < 1:
            Event_stoT = np.zeros(len(prob_eval))
            for i in range(0, len(prob_eval)):
                if prob_eval[i] < self.tsh[0]:
                    Event_stoT[i] = 1

            EventstoT_index = [j for j, i in enumerate(Event_stoT) if Event_stoT[j] == 1]
            self.tsh[0] = self.tsh[0] * 2

        EventsPR = np.zeros(len(self.events_real))
        EventsPR[EventstoT_index] = 1

        MatchEvent_p = np.subtract(EventsPR, self.events_real)
        No_EventsP_index = [j for j, i in enumerate(MatchEvent_p) if MatchEvent_p[j] == 0]
        No_EventsP = len(No_EventsP_index)
        No_EventsP_incorr = len(MatchEvent_p) - No_EventsP
        AccuracyP = np.divide(No_EventsP, len(MatchEvent_p))
        TP = No_EventsP
        index_FP = [j for j, i in enumerate(MatchEvent_p) if MatchEvent_p[j] == 1]
        FP = len(index_FP)
        index_FN = [j for j, i in enumerate(MatchEvent_p) if MatchEvent_p[j] == -1]
        FN = len(index_FN)
        Recall = np.divide(TP, np.sum([TP, FN]))
        Precision = np.divide(TP, np.sum([TP, FP]))
        F1_score = np.divide(2 * Precision * Recall, np.sum([Precision, Recall]))
        print(': No corr events: ' + str(No_EventsP))
        print(': No incorr events: ' + str(No_EventsP_incorr))
        print(': Accuracy: ' + str(AccuracyP))
        print(': Recall: ' + str(Recall))
        print(': Precision: ' + str(Precision))
        print(': F1-score: ' + str(F1_score))

        return EventstoT_index