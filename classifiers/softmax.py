import numpy as np
from random import shuffle
import matplotlib.pyplot as plt


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    nrm_p = np.zeros((num_train, num_classes))

    for i in range(num_train):
        # get ith row
        f = np.dot(X[i], W)
        f -= np.max(f)

        un_p = np.exp(f)
        un_p_row_sum = np.sum(un_p)
        nrm_p[i] = np.divide(un_p, un_p_row_sum)

        y_norm_p = nrm_p[i, y[i]]
        loss += -np.log(y_norm_p)

        nrm_p[i, y[i]] -= 1

    loss = loss / num_train  # average over number samples
    loss += 0.5 * reg * np.sum(W * W)

    nrm_p /= num_train
    dW = np.dot(X.T, nrm_p)
    db = np.sum(nrm_p, axis=0, keepdims=True)
    dW += reg * W  # don't forget the regularization gradient

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    num_train = X.shape[0]

    scores = softmax_get_UNnormalized_scores(X, W)
    normalized_scores = softmax_get_normalized_scores(scores)
    data_loss = softmax_get_data_loss(normalized_scores, num_train, y,reg,W)
    reg_loss = softmax_get_reg_loss(W, reg)
    loss =  data_loss + reg_loss

    dScores = get_dScores(normalized_scores, num_train,y)
    dW = softmax_get_dW(X,W,dScores,reg)
    return (loss,dW)

def softmax_get_UNnormalized_scores(X, W):
    """
    :param X: (N,D) inputs
    :param W: (D,C) outputs
    :return: norm_scores (N,C)

    each row adds to 1
    """
    return np.dot(X, W)

def softmax_get_normalized_scores(scores):
    '''

    :param scores: unnormalized scores
    :return: normalized scores
    '''
    scores -= np.max(scores)  # stability adjust, subtract the largest value from each row

    # raise all values to e to get unnormalized probabilities(up)
    un_p = np.exp(scores)

    # sum each row (N,1)
    un_p_row_sum = np.sum(un_p, axis=1).reshape(un_p.shape[0], 1)

    # divide each row element by sum of the row
    # each row now adds to 1 normalized probabilities (p_norm)
    return np.divide(un_p, un_p_row_sum)


def softmax_get_data_loss(norm_scores, num_train, y,reg, W1, W2=None):
    """
    :param norm_scores: (N,C) number examples, class scores for each ex
    :param num_train: (scaler)
    :param y:   (N,1) correct class score
    :return: loss (scaler)
    """
    # if no correct values then cannot calculate loss, or dW
    if y is None:
        raise ValueError("No correct values in y, cannot calculate loss and dW")

    # make (N,1) list of correct probs
    corect_logprobs  = norm_scores[range(num_train), y]

    # row adds up to 1 since I normalized fe_row_sum, so I dont have
    # to divide expression in () belowby np.sum(fe,axis=1)
    # notice that the loss is calculated ONLY using the correct loc
    tmp = -np.log(corect_logprobs )
    data_loss = (np.sum(tmp))/ num_train
    reg_loss = 0.5*reg*np.sum(W1*W1)
    if W2 is not None:
        reg_loss += 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    return loss  # average over number samples

def softmax_get_reg_loss(W, reg):
    """
    :param W:   (D,C)
    :param reg: scaler
    :return:    scaler total reg loss
    """
    return 0.5 * reg * np.sum(W * W)

def get_dScores(normalized_scores, num_train,y):
    """
    :param normalized_scores: (N,C)
    :param num_train: scaler
    :param y: (N,1) correct class scores
    :return: (N,C)
    """
    # start with normalized probs (each row adds to 1)
    dscores = np.copy(normalized_scores)

    # subtract 1 from correct class score
    dscores[range(num_train), y] -= 1

    # so increase the prob of non correct class leads
    # to decrease in correct class prob (all sum to 1)
    # which leads to increased loss (ln(smaller number) is larger than ln(1)=0)
    # in any case average aall these changes
    dscores /= num_train

    return dscores


def softmax_get_dW(X,W,dscores,reg = 0):
    '''
    :param X:
    :param W:
    :param dscores:
    :param reg:
    :return: the amount W should move
    '''

    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg * W  # don't forget the regularization gradient
    return dW



def softmax_loss_vectorized1(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = len(y)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    f = np.dot(X, W)  # (N,C)
    f -= np.max(f)  # stability adjust, subtract the largest value from each row

    # raise all values to e to get unnormalized probabilities(up)
    un_p = np.exp(f)

    # sum each row (N,1)
    un_p_row_sum = np.sum(un_p, axis=1).reshape(un_p.shape[0], 1)

    # divide each row elemnet by sum of the row
    # each row now adds to 1 normalized probabilities (p_norm)
    nrm_p = np.divide(un_p, un_p_row_sum)

    # make (N,1) list of correct probs
    y_norm_p = nrm_p[np.arange(num_train), y]

    # row adds up to 1 since I normalized fe_row_sum, so I dont have
    # to divide expression in () belowby np.sum(fe,axis=1)
    loss = -np.log(y_norm_p)
    loss = np.sum(loss) / num_train  # average over number samples
    loss += 0.5 * reg * np.sum(W * W)  # add in regularization

    # start with normalized probs (each row adds to 1)
    dscores = nrm_p

    # subtract 1 from correct class score
    dscores[range(num_train), y] -= 1

    # so increase the prob of non correct class leads
    # to decrease in correct class prob (all sum to 1)
    # which leads to increased loss (ln(smaller number) is larger than ln(1)=0)
    # in any case average aall these changes
    dscores /= num_train

    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg * W  # don't forget the regularization gradient

    return loss, dW
