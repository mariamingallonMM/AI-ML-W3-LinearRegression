"""
This code implements a linear regression per week 3 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.X for running on Vocareum
"""

# builtin modules
import os
#import psutil
import requests
import sys
import math


# 3rd party modules
import pandas as pd
import numpy as np


def RidgeRegression(X, y, l:int = 2):

    """
    This function implements the ℓ2 -regularized least squares linear regression algorithm (aka ridge regression) which takes the form:
    wRR=argminw(∥y−Xw∥^2 ) + λ * ∥w∥^2.
    
    Inputs:
    y : vector 'y' from the split dataset
    X : matrix 'X' from the split dataset
    l : an arbitray value for lambda "λ"
    
    Outputs:
    wRR : calculated as expressed above
    """
    n, m = X.shape
    I = np.identity(m)
    cov_matrix = np.linalg.inv(np.dot(X.T, X) + l * I)
    wRR = np.dot(np.dot(cov_matrix, X.T), y)
    #_, S, _ = np.linalg.svd(X)
    #deg_f = np.sum(np.square(S) / (np.square(S) + l))

    return wRR

def update_posterior(X, y, xx_old, xy_old, l:int = 2, sigma2:int = 3):
    n, m = X.shape
    I = np.identity(m)

    xx_old = np.dot(X.T, X) + xx_old  # XoXoT + XiXiT 
    xy_old = np.dot(X.T, y) + xy_old  # Xoyo + Xiyi

    new_cov = np.linalg.inv(l * I + (1 / sigma2) * xx_old) # new covariance Σ, captures uncertainty about w as Var[wRR]
    new_mean = np.dot((np.linalg.inv(l * sigma2 * I + xx_old)),xy_old) # new µ
    
    return new_cov, new_mean, xx_old, xy_old

def activeLearning(X, y, X_test, l:int = 2, sigma2:int = 3):

    n, m = X.shape
    
    I = np.identity(m)

    X_indexes = []

    # set up an xx_old and xy_old to zero to start with
    xx_old = np.zeros((m, m))
    xy_old = np.zeros(m)

    new_cov, new_mean, xx_old, xy_old = update_posterior(X, y, xx_old, xy_old, l, sigma2)

    #Select the 10 data points to measure as per the assignment requirements
    for i in range(10):
        # Pick x0 for which sigma20 is largest
        
        cov_matrix = np.dot(np.dot(X_test, new_cov), X_test.T)
        row_largest = np.argmax(np.diagonal(cov_matrix))

        # update x and y values
        # first reset indexes for X_test dataframe so you can call it from its row index number using 'row_largest'
        #X_test = X_test.reset_index(drop=True) #this line is not needed when running in Vocareum presumuably because the X_data.csv does not include an index column.
        # then get x0 and pass it onto the X matrix
        #x0 = X_test.iloc[[row_largest]].to_numpy()[0]
        # use the following when running on Vocareum
        x0 = X_test[row_largest] #.to_numpy()[0]

        #X.loc[row_largest] = x0
        
        # use the following when running on Vocareum 
        X[row_largest] = x0
        
        # calculate y0 and update the y vector
        y0 = np.dot(X, new_mean)
        y = y0

        #row, _ = X_test.shape
        #X_index = list(range(row))[row_largest]
        X_indexes.append(row_largest)

        # Remove x0
        #X_test = X_test.drop(index = row_largest)
        
        # use the following when running on Vocareum 
        X_test = np.delete(X_test, (row_largest), axis=0)        
        # Update the posterior distribution
        new_cov, new_mean, xx_old, xy_old = update_posterior(X, y, xx_old, xy_old, l, sigma2)

    # Create 1-based indexes
    X_indexes = [i + 1 for i in X_indexes]

    return X_indexes

def main():
    """
    ## $ python3 hw1_regression.py lambda sigma2 X_train.csv y_train.csv X_test.csv
    """
    # get inputs into local variables
    lambda_input = int(sys.argv[1])
    sigma2_input = float(sys.argv[2])
    X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
    y_train = np.genfromtxt(sys.argv[4])
    X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

    #### Part 1
    wRR = RidgeRegression(X_train, y_train, lambda_input)
    # write the output csv file with the wRR values
    np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, fmt='%1.2f', delimiter="\n")

    #### Part 2
    X_indexes = activeLearning(X_train, y_train, X_test, lambda_input, sigma2_input)
    # write the output csv file
    np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", X_indexes, fmt='%.d', delimiter="\n")
    
if __name__ == '__main__':
    main()
