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

    rows = df.shape[0]
    cols = df.shape[1]
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
    #wRR = np.argmin((y - np.dot(X, w))**2) + l * w**2

    xtranspose = np.transpose(X)
    xtransx = np.dot(xtranspose, X)
    if xtransx.shape[0] != xtransx.shape[1]:
        raise ValueError('Needs to be a square matrix for inverse')
    lamidentity = np.identity(xtransx.shape[0]) * l
    matinv = np.linalg.inv(lamidentity + xtransx)
    xtransy = np.dot(xtranspose, y)
    wRR = np.dot(matinv, xtransy)
    _, S, _ = np.linalg.svd(X)
    deg_f = np.sum(np.square(S) / (np.square(S) + l))

    return wRR, deg_f

# implementation of gradient descent algorithm  
def gradientDescent(x, y, theta, alpha, num_iters, c):
    # get the number of samples in the training
    m = x.shape[0]
    
    for i in range(0, num_iters):
        
        # find linear regression equation value, X and theta
        z = np.dot(x, theta)
        
        # get the sigmoid of z
        h = sigmoid(z)
 
        # calculate the cost function, log loss
        #J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1-h)))
        
        # let's add L2 regularization
        # c is L2 regularizer term
        J = (-1/m) * ((np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1-h))) + (c * np.sum(theta)))
        
        # update the weights theta
        theta = theta - (alpha / m) * np.dot((x.T), (h - y))
   
    J = float(J)
    return J, theta


def write_csv(filename, a, b, c):
        # write the outputs csv file
        filepath = os.path.join(os.getcwd(),'datasets','out', filename)
        df_a = pd.DataFrame(a)
        df_b = pd.DataFrame(b)
        df_c = pd.DataFrame(c)
        df = pd.concat([df_a, df_b, df_c], axis = 1, ignore_index = True)
        #dataframe = df.rename(columns={0:'alpha',1:'number_of_iterations',2:'b_0', 3:'b_age',4:'b_weight'})
        df.to_csv(filepath, index = False, header = False)
        return print("New Outputs file saved to: <<", filename, ">>", sep='', end='\n')


def main():
    """
    ## $ python3 problem3.py input3.csv output3.csv
    """
    #take string for input data csv file
    #in_data = str(sys.argv[1])
    in_data = 'forestfires.csv'
    #take string for output data csv file
    #out_data = str(sys.argv[2])
    out_data = 'output3.csv'

    df = get_data(in_data)
    # split the dataset
    df_X_train, df_X_test, df_y_train, df_y_test = split_data(df, ratio = 0.7)

    vis_data(df_X_train, df_y_train)

    # TODO: function for the following iteration?
    # solve the linear regression (ridge regression) for a varying lambda parameter
    wRR_list = []
    deg_f_list = []

    # TODO: resolve issue with dimensions of dataframes, dotproduct, etc
    for l_i in range(0,5000,1):
        wRR, deg_f = RidgeRegression(df_X_train,df_y_train,l_i)
        wRR_list.append(wRR)
        deg_f_list.append(deg_f)

    #TODO: plot results
    
if __name__ == '__main__':
    main()
