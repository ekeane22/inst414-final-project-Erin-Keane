
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging 
import mylib 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

#transformed_directory = Path('../data/transformed')
#Extracted_directory = Path('../data/extracted')

'''
To evaluate my model, i am using mean squared error and r squared 

'''

# I would read in the csv from the model.py from outputs/

def evalaute(model, X_test, y_test): 
    '''
    Evaluates the linear regression model. 
    
    Args: 
        model: the trained linear regression model
        X_train (DF): features for training 
        y_train(series): target variable for training 
    
    Returns: 
        dict
    
    '''
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

return #mean squared error and r2 

#save the results to outputs/
    