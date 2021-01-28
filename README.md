# AI-ML-W3-LinearRegression
ColumbiaX CSMM.102x Machine Learning Course. Week 3 Assignment on Linear Regression.


## Instructions
This assignment has two parts. The aim is to write a function that implements both.

**PART 1:** In this part we will implement the ℓ2 -regularized least squares linear regression algorithm (aka ridge regression) which takes the form:

wRR=argminw∥y−Xw∥^2+λ∥w∥2. 

Write the code that takes in data 'y'  and 'X' and outputs 'wRR' for an arbitrary value of  'λ'.

**PART 2:** In the same code, we will also implement the ***'active learning procedure'***. The assignment provides an arbitrary setting of  'λ'  and  'σ2'  and asks that you provide the first 10 locations you would measure from a set D={x}  given a set of measured pairs (y,X). Please look over the slides carefully to remind yourself about the sequential evolution of the sets D and (y,X).

More details about the inputs we provide and the expected outputs are given below.

## What you need to submit

## Execute the program
The following command shall execute your program
$ python hw1_regression.py lambda sigma2 X_train.csv y_train.csv X_test.csv

Note the following:
- The name of the train and testing X and y datasets. 
- The main .py file shall be named 'hw1_regression.py'.
- The value of 'lambda' ("λ") above will be a non-negative integer (any non-negative integer that is chosen by the reviewers when running the code).
- The value of sigma2 ("σ2") above will be an arbitrary positive number.

The values of lambda and sigma2 will be input as strings. They are converted to numbers in the code for it to work. Note that all numbers should be double-precision floating-point format.

The csv files that we will input into your code are formatted as follows:

- X_train.csv: A comma separated file containing the covariates. Each row corresponds to a single vector  xi . The last dimension has already been set equal to 1 for all data.
- y_train.csv: A file containing the outputs. Each row has a single number and the i-th row of this file combined with the i-th row of "X_train.csv" constitutes the training pair  (yi,xi) .
- X_test.csv: This file follows exactly the same format as "X_train.csv". No response file is given for the testing data.

## Expected Outputs from the program
When executed, the code writes the output for both PART 1 & 2 to the files listed below. Note the formatting instructions given below. For an arbitrary chosen value of "λ" and "σ2", you will create the following two files containing:

- wRR_[lambda].csv: A file where the value in each dimension of the vector wRR is contained on a new line. This file corresponds to your output for PART 1 above.
- active_[lambda]_[sigma2].csv: A comma separated file containing the row index of the first 10 vectors you would select from X_test.csv starting with the measured values in X_train.csv and y_train.csv. Please make sure your indexing starts at 1 and not at 0 (i.e., the first row is index 1). This file should contain one line with a "," separating each index value. This file corresponds to your output for PART 2 above.

For example, if  λ=2  and  σ2=3 , then the output files the code will create will be named: 
"wRR_2.csv" and "active_2_3.csv"

If your code then learns that w = [3.2; -3.6; 1.4; -0.7], then wRR_2.csv should look like:

3.2
-3.6
1.4
-0.7

If  the first 10 index values you would choose to measure are 724, 12, 109, 42, 23, 96, 342, 594, 123, 414, then active_2_3.csv should look like:

724,12,109,42,23,96,342,594,123,414

## Note on Correctness

Please note that for both of these problems, there is one and only one correct solution. Therefore, we will grade your output based on how close your results are to the correct answer. We strongly suggest that you test out your code on your own computer before submitting. The UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/) has a good selection of datasets for regression. While you still may not have the ground truth, you can build confidence that the outputs of your code are reasonable. For example, you can verify that your vector  wRR  makes reasonable predictions and that your 10 selected measurement indexes are all unique.

## Notes on data repositories
The following datasets have been selected from the UCI Machine Learning Repository for use and testing of the code written for this assignment:

- [Forest Fires Data Set](http://archive.ics.uci.edu/ml/datasets/Forest+Fires). This is a difficult regression task, where the aim is to predict the burned area of forest fires, in the northeast region of Portugal, by using meteorological and other data (see details [here](http://www.dsi.uminho.pt/~pcortez/forestfires)).
- [Wine Quality Data Set](http://archive.ics.uci.edu/ml/datasets/Wine+Quality). Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests (see [Cortez et al., 2009](http://www3.dsi.uminho.pt/pcortez/wine/)).


## Citations & References

- [Forest Fires Data Set](http://archive.ics.uci.edu/ml/datasets/Forest+Fires) by P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimaraes, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9.
- [Wine Quality Data Set](http://archive.ics.uci.edu/ml/datasets/Wine+Quality) by P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.



