#!/usr/bin/env python
''' TwoLayer_Neural_Sigmoid_Softmax_BintoDec.py backprop demo with softmax
binary to decimal neural net decoder
1 hidden layer using sigmoids
1 outputlayer using softmax

takes binary digits and converts to decimal equiv using an 8 output softmax
NOTE:  This is an overtrained network that will only work with the supplied data it
will not generalize so make sure you give it complete training vectors (ie 3 bits
include all 8 possible combonations'''

__author__   = "Keith Perkins"
__credits__  = "basic-python-network  on iamtrask"

import numpy as np
import classifiers.softmax as sm

class TwoLayer_Neural_Sigmoid_Softmax_BintoDec:
    def __init__(self, reg = 1e-5,hiddenLayerSize = 4,numbTrainIterations = 100000):
        self.hiddenLayerSize = hiddenLayerSize
        self.numbTrainIterations = numbTrainIterations
        self.reg = reg

    def predict(self,X):
        '''
        Vectorized: predicts  class given X (if X=010 will return 2)
        :param X:
        :return: Class
        '''
        #output of hidden layer using sigmoid
        self.l1 = 1/(1+np.exp(-(np.dot(X,self.W0))))

        #get the normalized output probabilities for each class
        scores = sm.softmax_get_UNnormalized_scores(self.l1,self.W1)
        self.norm_scores = sm.softmax_get_normalized_scores(scores)

        return np.argmax(self.norm_scores, axis=1)

    def fit(self,X,y):
        '''
        :param X: (N,D)  N = number examples, D=number bits
        :param y: (N,1)  array of correct classes (ie 010 = 2)
        :return: loss  total loss accross all inputs
        '''
        num_train = X.shape[0]
        num_inputs = X.shape[1]
        num_outputs = 2**num_inputs

        #must remember the trained weights
        #these are 0 centered
        self.W0 = 2*np.random.random((num_inputs,self.hiddenLayerSize)) -1
        self.W1 = 2*np.random.random((self.hiddenLayerSize,num_outputs)) -1

        #log every 100th loss so we can plot loss later
        rowskip = self.numbTrainIterations/100
        self.loss=[]

        #training
        for j in range (self.numbTrainIterations):

            self.predict(X)

            # if j==self.numbTrainIterations-1:
            #     print(str(self.norm_scores))

            #calculate the total loss over all classes
            if j%rowskip == 0:
                self.loss.append(sm.softmax_get_data_loss(self.norm_scores, num_train, y,self.reg, self.W0, self.W1))

            #get the softmax derivative
            norm_scores_delta =  sm.get_dScores(self.norm_scores, num_train,y)

            #find change to W1(negate it so we decrease loss)
            #this is backprop (dL2/dx)(W.T*(dL1/dx))
            l1_delta = norm_scores_delta.dot(self.W1.T)*(self.l1*(1-self.l1))

            #why the l1 and X?
            self.W1 -= self.l1.T.dot(norm_scores_delta)
            self.W0 -= X.T.dot(l1_delta)

        return self.loss





