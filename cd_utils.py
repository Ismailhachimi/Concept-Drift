#!/usr/bin/env python

"""
This module includes functions for Concept Drift Detection with a data iterator as a Stream source.
"""

__author__  = "Ismail El Hachimi & Mathieu Vandecasteele"
__title__   = "Concept Drift Detection Through Resampling"

# Dependencies

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import numpy as np
from time import time, sleep

np.random.seed(42)

# Data Stream Class

class DataStream:
    
    def __init__(self, X, y, size):
        self.iterator = iter(zip(X, y))
        self.buffer = []
        self.size = size
        self.xshape = X.shape
        self.yshape = y.shape

    def __iter__(self):
        return self

    def __next__(self):
        tmp = (np.zeros((self.size, self.xshape[1])), 
               np.zeros(self.size))

        for i in range(self.size):
            if self.buffer:
                tmp_iter = self.buffer.pop()
            else:
                tmp_iter = next(self.iterator)
            tmp[0][i,:] = tmp_iter[0]
            tmp[1][i] = tmp_iter[1]   
        return tmp

    def has_next(self):
        if self.buffer:
            return True
        try:
            self.buffer = [next(self.iterator)]
        except StopIteration:
            return False
        else:
            return True

# Concept Drift Functions

def get_data_stream(data_stream, X=None, y=None):
    if X is None and y is None:
        return data_stream
    else:
        return np.concatenate((X, data_stream[0]), axis=0), np.concatenate((y, data_stream[1]), axis=0)

def concept_drift_scheme(window_size, permut, cd_size, significance_rate, data_stream):

    t_ = 0 
    k = window_size  
    D = [] 
    X, y = get_data_stream(data_stream.__next__(), None, None)
    times = []
    i = 1
    
    while(data_stream.has_next()):
        print("############# STREAM N° {} #############".format(i))
        i += 1
        time_s = time()
        X, y = get_data_stream(data_stream.__next__(), X, y)
        Sx_ord, Sx_ord_t, Sy_ord, Sy_ord_t = train_test_split(X[t_:, :], y[t_:], test_size=window_size,
                                                              shuffle=False)
        if detect_concept_drift((X[t_:, :], y[t_:]), (Sx_ord, Sy_ord), (Sx_ord_t, Sy_ord_t), 
                                permut, cd_size, significance_rate, window_size):
            print("\nA CONCEPT DRIFT HAS BEEN DETECTED at k = {}\n".format(k))
            t_ = k
            D.append(k)
        k += window_size
        time_s = time() - time_s
        times.append(time_s)
        print("Took : "+str(time_s))
    return D, times
        
def detect_concept_drift(data, S_ord, S_ord_t, permut, cd_size, significance_rate, window_size):
    Rord = empirical_risk(S_ord, S_ord_t)
    Rs = []
    S = []
    S_t = []
    for i in range(permut):
        X, X_t, y, y_t = train_test_split(data[0], data[1],  test_size=window_size, shuffle=True)
        S.append((X, y))
        S_t.append((X_t, y_t))
        Rs.append(empirical_risk(S[-1], S_t[-1]))        
    return TEST(Rord, Rs, cd_size, significance_rate)
         
def empirical_risk(S, S_t):
    hyperparams = {'kernel': 'rbf', 
                   'C': 5, 
                   'gamma': 0.05}
    model = SVC(**hyperparams)
    model.fit(S[0], S[1])
    return model.score(S_t[0], S_t[1])
                  
def TEST(Rord, Rs, cd_size, significance_rate): 
    nb_detected = 1
    for i in range(len(Rs)):
        nb_detected += ((Rord - Rs[i]) <= cd_size)*1    
    tmp = nb_detected/(len(Rs) + 1)
    print("Well "+str(tmp))
    if tmp <= significance_rate:
        return True
    else:
        return False