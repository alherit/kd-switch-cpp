# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:11:05 2019

@author: alheritier
"""
from pykds import KDSForest
import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

dim = data.data.shape[1]
alpha_label= len(np.unique(data.target))

m =  KDSForest(ntrees=1,seed=123,dim=dim,alpha_label=alpha_label,ctw=False, theta0=[])
nll = 0

for point,label in zip(data.data, data.target):
    log_probs = m.predict_log2_proba(point=point)
    obs_prob = 2 ** log_probs[label]
    nll += - log_probs[label]
    print("prob assigned to observed symbol ", label, " : ", 2 ** log_probs[label] )
    m.update(point=point,label=label)

nll /= data.data.shape[0]
print("Normalized Log Loss: ", nll)    

