'''
MODEL IS FIRST, THEN EVALUATE 

Model: 
    multiple linear Regression - adding complexity and it needs certain types of data 
or simple linear    

'''


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


#logging.basicConfig(filename='regression.log', filemode='w', level=logging.DEBUG)
#logger = logging.getLogger(__name__)

loaded_directory = Path('../data/loaded')

outputs_directory = Path('../data/outputs')

def load_from_loaded(loaded_directory): 
    '''
    Loads the two csv dataframes from the extract folder from the data directory. 
    
    Args: 
        loaded_directory (Path): Path to the folder containing the dataframes from extracted.py.
        
    Returns: 
        tuple: Contains the new merged county22 and flight22 DataFrames called wf.
    '''
    
    #try: the line under should be indented 
    wf = pd.read_csv(loaded_directory / 'wf.csv')
        #logger.info('Data successfully loaded from CSV.')
    #except Exception as e: 
        #logger.error(f'Error loading the data: {e}')
        #raise
    return wf 

def preprocess(wf): 
    '''
    Preprocess the data for regression analysis. 
    
    Args: 
        wf (DF): The merged dataframe with weather and flight information. 
        
    Returns: 
        Df and series: the features and target variable 
    
    '''
    
    wf.fillna(0, inplace=True)
    
    #feature engineering 
    wf['season'] = pd.to
    
    #defining features and target variables 
    features = ['FlightDate','OriginCityName', 'Cancelled', 'CancellationCode', 'WeatherDelay']
    target = 'EventType'
    
    X = wf(features)
    y = wf(target)
    
    return X, y

def train_model(X_train, y_train): 
    '''
    Train the model. 
    
    Args: 
        X_train (DF): features for training 
        y_train(series): target variable for training 
    
    Returns: 
        model: trained linear regression model 
    '''
    model = LinearRegression() 
    model.fit(X_train, y_train)
    return model 



    
#handle missing values 
#cancelled 
#cancellation code b = weather 
#carrierdelay 
#weather delay 
#NASdelay 
#security delay 
#late aircraft delay 

# no categorical variables 
#either encode them or drop them.... encode them??
# test train split 
#from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.linear_model import LogisticRegression as lr
# the train test split = regression sklearn
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

#accuracy isnt the most important

# Feature engineering = which columns are important and then drop the rest of them 
#sometiems the more features the worse your model is 
#pitfall to which features are ideal and how they interact 


