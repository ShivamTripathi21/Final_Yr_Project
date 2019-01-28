#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 12:47:33 2018

@author: shivam
"""
"""
Implementing NearestNeighbor Classifier
Image Classifier

we r gonna use cifr-10 data set for classification
"""

import numpy as np
from Desktop.DL_API.data_set.data_util import *

class NearestNeighbor:
    def __init__(self):
        pass
    
    #Memorize training data
    def train(self,X,y):
        """ X is N x D where N is number of images and D is 32*32*3, y is 1 dimension of size N"""
        self.x = X
        self.y = y
    
    def predict(self, X):
        """ X is N x D where each ro is an example we wish to predict label for"""
        num_img = X.shape[0]
        ypred = np.zeros(num_img, dtype = self.y.dtype)
        
        #loop over all test rows
        for i in xrange(num_img):
            #calculate l1 distance
            distances = np.sum(np.abs(self.x - X[i,:]), axis = 1)
            min_index = np.argmin(distances)
            ypred = self.y[min_index]
        return ypred
    

"""
    training time O(1)
    testing time O(n)
"""    

"""checking data accessiblity"""

x, y = load_cifar_batch("data_batch_1")
print(x.shape)


        
