"""
This code implements a linear regression per week 3 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.7
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
import plotly.graph_objects as go


def vis_data(data_x, data_y):

    fig = go.Figure()

    for col in data_x.columns:
        fig.add_trace(go.Scatter(x=data_x[col], y=data_y,
                                 mode='markers',
                                 name="{}".format(col)))
        
    return fig.show()


def vis_regression_(X, y):

    #TODO: fix dimensionality of dataframe plotted per column as above for actual curve

    fig = go.Figure()

    for col in data_x.columns:
        fig.add_trace(go.Scatter(x=data_x[col], y=data_y,
                                 mode='markers',
                                 name="{}".format(col)))
    return fig.show()


def get_data(source_file):

# Define input and output filepaths
    input_path = os.path.join(os.getcwd(),'datasets','in', source_file)

    # Read input data
    df = pd.read_csv(input_path)
       
    return df


def split_data(df, ratio:float = 0.7):

    """
    Splits the data set into the training and testing datasets from the following inputs:

    df: dataframe of the dataset to split
    ratio : percentage by which to split the dataset; ratio = training data / all data;
    e.g. a ratio of 0.70 is equivalent to 70% of the dataset being dedicated to the training data and the remainder (30%) to testing.

    Outputs: X_train, y_train, X_test, y_test
   
    X_train: Each row corresponds to a single vector  xi . The last dimension has already been set equal to 1 for all data.
    y_train: Each row has a single number and the i-th row of this file combined with the i-th row of "X_train" constitutes the training pair (yi,xi).
    X_test: The remainder of the dataset not included already in the X_train dataframe. Same format as "X_train".
    y_test: The remainder of the dataset not included already in teh y_train dataframe. Same format as "y_train".

    """

    rows, cols = df.shape
    rows_split = int(ratio * rows)

    # split the dataset into train and test sets
    # drop last column of X which will become 'y' vector
    # TODO: add a column of '1' to the X matrix
    df_X_train = df[df.columns[:-1]].loc[0 : rows_split]
    df_X_test = df[df.columns[:-1]].loc[rows_split : rows]

    # get the last column of X as the 'y' vector and split it into train and test sets
    df_y_train = df[df.columns[cols - 1]].loc[0 : rows_split] 
    df_y_test = df[df.columns[cols - 1]].loc[rows_split : rows]

    return df_X_train, df_X_test, df_y_train, df_y_test


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
    deg_f : degrees of freedom as a function of lambda
    """
    n, m = X.shape
    I = np.identity(m)
    cov_matrix = np.linalg.inv(np.dot(X.T, X) + l * I)
    wRR = np.dot(np.dot(cov_matrix, X.T), y)
    _, S, _ = np.linalg.svd(X)
    deg_f = np.sum(np.square(S) / (np.square(S) + l))

    return wRR, cov_matrix, deg_f

def update_posterior(X, y, X_old, y_old, l:int = 2, sigma2:int = 3):
    n, m = X.shape
    I = np.identity(m)

    X_old = np.dot(X.T, X) + X_old
    y_old = np.dot(X.T, y) + y_old

    new_cov = np.linalg.inv(l * I + (1 / sigma2) * X_old)
    new_mean = (np.linalg.inv(l * sigma2 * I + X_old)).dot(y_old)

    return new_cov, new_mean, X_old, y_old

def activeLearning(X, y, X_test, l:int = 2, sigma2:int = 3):

    n, m = X.shape
    I = np.identity(m)

    X_indexes = []

    X_old = np.zeros((m, m))
    y_old = np.zeros(m)

    new_cov, new_mean, X_old, y_old = update_posterior(X, y, X_old, y_old, l, sigma2)

    #Select the 10 data points to measure as per the assignment requirements
    for i in range(10):
        # Pick x0 for which sigma20 is largest
        cov_matrix = np.dot((np.dot(X_test, new_cov)), X_test.T)
        row_largest = np.argmax(np.diagonal(cov_matrix))

        # update x and y values
        X_test = X_test.reset_index(drop=True)
        X = X_test.iloc[[row_largest]].to_numpy()[0]
        y = np.dot(X, new_mean)

        #row, _ = X_test.shape
        #X_index = list(range(row))[row_largest]
        X_indexes.append(row_largest)

        # Remove x0
        X_test = X_test.drop(index = row_largest)

        # Update the posterior distribution
        # TODO: fix error here: 'not enough values to unpack because X is an 1D array'; review the conversion of X onto a 1D array above
        new_cov, new_mean, X_old, y_old = update_posterior(X, y, X_old, y_old, l, sigma2)

    # Create 1-based indexes
    X_indexes = [i + 1 for i in X_indexes]

    return X_indexes


def write_csv(filename, a):
        # write the outputs csv file
        filepath = os.path.join(os.getcwd(),'datasets','out', filename)
        df = pd.DataFrame(a)
        df.to_csv(filepath, index = False, header = False)
        return print("New Outputs file saved to: <<", filename, ">>", sep='', end='\n')

def main():
    """
    ## $ python3 hw1_regression.py lambda sigma2 X_train.csv y_train.csv X_test.csv
    """
    #take string for input data csv file
    #in_data = str(sys.argv[1])
    #uncomment the following when ready to bring in lambda_input and sigma2_input
    #lambda_input = int(sys.argv[1])
    #sigma2_input = float(sys.argv[2])
    
    #in_data = 'forestfires.csv'
    in_data = 'winequality-red.csv'

    df = get_data(in_data)
    # split the dataset
    df_X_train, df_X_test, df_y_train, df_y_test = split_data(df, ratio = 0.7)
    
    vis_data(df_X_train, df_y_train)

    write_csv('X_train.csv', df_X_train)
    write_csv('y_train.csv', df_y_train)
    write_csv('X_test.csv', df_X_test)

    #uncomment the following when ready
    #X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
    #y_train = np.genfromtxt(sys.argv[4])
    #X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

    #### Part 1
    lambda_input = 2
    wRR, cov_matrix, deg_f = RidgeRegression(df_X_train,df_y_train,lambda_input)
    # write the output csv file with the wRR values
    np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n")

    #### Part 2
    sigma2_input = 3
    X_indexes = activeLearning(df_X_train, df_y_train, df_X_test, lambda_input, sigma2_input)
    # write the output csv file
    np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",")
    
if __name__ == '__main__':
    main()
