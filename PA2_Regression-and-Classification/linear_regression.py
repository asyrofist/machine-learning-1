"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    w = np.reshape(w, [len(w), 1])
    y = np.reshape(y, [len(y), 1])
    Xw = np.matmul(X, w)
    err = np.sum((Xw - y) ** 2) / len(X)
    #####################################################
    
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    #	TODO 2: Fill in your code here #
    #invX = np.linalg.inv(np.matmul(X.transpose(), X))
    #IX = np.matmul(invX, X.transpose())
    #w = np.matmul(IX, y.reshape([len(y), 1]))
    
    w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    #####################################################		

    return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    I = np.dot(X.T, X)
    eig = np.linalg.eigvals(I)
    while min(np.abs(eig)) < 10 ** -5:
        I += .1 * np.eye(len(I))
        eig = np.linalg.eigvals(I)
    w = np.dot(np.linalg.inv(I), np.dot(X.T, y))
    
    #####################################################

    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
    D = X.shape[1]
    s = np.eye(D)
    w = np.dot(np.linalg.inv(lambd * s + np.dot(X.T, X)), np.dot(X.T, y))

  #####################################################		
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    bestlambda = None
    MSEmin=np.inf
    for i in np.arange(-19, 20):
        l=10.0**i
        w=regularized_linear_regression(Xtrain, ytrain, l)
        mse=mean_square_error(w, Xval, yval)
        if mse<MSEmin:
            MSEmin=mse
            bestlambda=l
    #####################################################		

    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    mapped_X=np.array(X)
    if power<2:
        return X
    for i in np.arange(2,power+1):
        powerX=np.power(X,i)

        mapped_X=np.concatenate((mapped_X, powerX),axis=1)
    X=mapped_X
    #####################################################		
    
    return X


