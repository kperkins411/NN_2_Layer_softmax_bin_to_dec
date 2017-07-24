#!/usr/bin/env python
''' test_BintoDec.py backprop demo with softmax
simple tests on a simple neural network
'''

from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from classifiers.TwoLayer_Neural_Sigmoid_Softmax_BintoDec import TwoLayer_Neural_Sigmoid_Softmax_BintoDec


class TestBintoDec(TestCase):
    MIN_ACCEPTABLE_ERROR = 1e-1
    myclass = None

    @classmethod
    def setUpClass(cls):
        TestBintoDec.myclass = TwoLayer_Neural_Sigmoid_Softmax_BintoDec()

    @classmethod
    def setUp(self):
        self.X= np.array([ [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1] ])
        self.y = np.array([0,1,2,3,4,5,6,7])

    def test_fit(self):
        loss = TestBintoDec.myclass.fit(self.X,self.y)
        loss_size = len(loss)
        plt.scatter(range(loss_size), loss)
        plt.ylabel('Loss')
        plt.xlabel('Measurement')
        plt.show()
        self.assertLess(loss[loss_size-1],self.MIN_ACCEPTABLE_ERROR,"Error %s is larger than min %s"%(str(loss),str(self.MIN_ACCEPTABLE_ERROR)) )

    def test_predict(self):
        X= np.array([[1,1,0]])
        y= np.array([6])
        pred = TestBintoDec.myclass.predict(X)
        self.assertEqual(y[0], pred, "Expected %s got %s"%(y[0], pred))
