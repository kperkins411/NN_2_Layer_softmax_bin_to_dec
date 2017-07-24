import numpy as np
from unittest import TestCase
from unittest import main
from classifiers.softmax import softmax_loss_naive
from classifiers.softmax import softmax_loss_vectorized
from classifiers.softmax import softmax_loss_vectorized1


#see http://cs231n.github.io/linear-classify/#svmvssoftmax for these sample matrices
def getW():
    #CxD
    return np.array([[.01,-.05,.1,.05,0],[.7,.2,.05,.16,.2],[0,-.45,-.2,.03,-.3]])
def getX():
    #D,N
    return np.array([[-15,-15],[22,22],[-44,-44],[56,56],[1,1]])
def gety():
    return np.array([2,2])



def getW_T():
    #DxC
    return getW().transpose()
def getX_T():
    #(N,D)
    return getX().transpose()
def gety_T():
    #(N,)
    return gety().transpose()

class TestSoftmax_loss_naive(TestCase):

    def setUp(self):
        self.W = getW_T()
        self.X= getX_T()
        self.y = gety_T()   #same as gety()
        self.reg = 1e-3     #regularization strength


    def test_If_Dotted_Transposes_Are_Equivelant(self):
        #does W.X = (XT.WT).T?
        W=getW()
        X=getX()
        F1 = np.dot(W,X)
        F2 = np.dot(X.transpose(), W.transpose()).transpose()
        self.assertTrue((F1==F2).all()," W.X != (XT.WT).T")


    def test_softmax_loss_naive(self):
        loss,dw = softmax_loss_naive(self.W, self.X, self.y, self.reg)
        pass

    def test_both_softmax_loss_vectorized(self):
        loss,dw = softmax_loss_vectorized(self.W, self.X, self.y, self.reg)
        loss1,dw1 = softmax_loss_vectorized1(self.W, self.X, self.y, self.reg)
        self.assertLess((loss-loss1),1e-3,"Losses are off")
        self.assertLess( (dw-dw1).sum(),1e-3,"Vectors are off")



