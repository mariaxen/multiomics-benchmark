#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:03:50 2020

@author: maria
"""

import pickle
from random import sample
import matplotlib.colors as colors
import numpy as np


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def pickle_dataset(path, DF, groups, responses):
    #Pickle them for future use
    with open(path+'/DF.pkl', 'wb') as f:  
        pickle.dump(DF, f)

    with open(path+'/groups.pkl', 'wb') as f:  
        pickle.dump(groups, f)

    with open(path+'/responses.pkl', 'wb') as f:  
        pickle.dump(responses, f)
        
        
def sample_features(DF, omic_names, n_samples):

    responses = []
    
    for predictor_index in range(len(omic_names)):

        #Get your response dataset
        response = sample([col for col in DF if col.startswith(omic_names[predictor_index])], n_samples)
        responses.append(response)
    
    return responses
