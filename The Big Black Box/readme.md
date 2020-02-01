# The Bigger Blacker Box by a Lazy Data Scientist

While making predictive models we come across the same process everywhere. Clean data, encode categorical variables, impute missing values and finally choose an algorithm. But its not that simple, is it? Too many ways to - analyse and clean data, encode categorical features, impute values and on top of that so many algorithms with just as many parameters. How do you find the best solution?

The data cleaning generally is domain oriented. Feature selection is majorly done by the algorithm – considering how far we have come with the algorithms. But if we can choose features automatically, can we not have a similar method to choose the encoders and the imputers? Just think about how many tests, p-values and bonferroni corrections we will we be skipping. Lately a lot of auto-ml tools have popped up which point to people wanting to make use of this black box more and more everyday.

I decided to do something similar. I programmed random selection for encoders, imputers and modelling algorithms to see if using more macro hyper-parameters could get me some useful information about modelling the data in order to make predictions for the unknowns. This is my 1st step to having my own auto-ml someday maybe. This project is the first piece of a huge puzzle that I will put together in time.

#### The repository contains
  - B3.py: Stand alone python code that contains all the functions within the same file (Still need to add command prompt arguements and object classes)
  - csv files for the data to be used for this example
  - Example summary file that was generated by the program

# Methodology
  - Data is first read into pandas dataframes and cleaned. This function for now is manual and may be skipped. But make sure that the number of columns in both train and test data are same
  - Encoder combinations are created selecting random encoders for each categorical column
  - Encoder combination are used to encode the data
  - Imputer combinations are created selecting random imputers for each column
  - Imputer combination are used to impute the data
  - Data is sent to the modeller that selects the modelling algorithm, which randomly selects an algorithm and calculates validation error from a peice of data from train data we set aside right from the beginning
  - The results are cross validated with multiple validation data segments and summarised in a csv file
  - 
### Tech
B3 uses a number of open source projects to work properly:

* [scikit-learn/scikit-learn] - https://github.com/scikit-learn/scikit-learn
* [pandas-dev/pandas] - https://github.com/pandas-dev/pandas
* [yzhao062/pyod] - https://github.com/yzhao062/pyod
* [numpy/numpy] - https://github.com/numpy/numpy
* [catboost/catboost] - https://github.com/catboost/catboost
* [dmlc/xgboost] - https://github.com/dmlc/xgboost
* [microsoft/LightGBM] - https://github.com/microsoft/LightGBM
* [scikit-learn-contrib/categorical-encoding] - https://github.com/scikit-learn-contrib/categorical-encoding
* [python/cpython] - https://github.com/python/cpython