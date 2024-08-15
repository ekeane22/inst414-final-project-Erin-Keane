'''
Bar Charts, Line Graph, Histograms, Scatte plots.

Can be exploratory and rough. 

I saw a picture of a scatter_geo based on clusers, might try. 

I think im going to revise and build different dataframes 
based on the graph I want to make... Maybe.  
'''

import plotly 
import bokeh 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
from pathlib import Path
import logging 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

logging.basicConfig(
    filename='vis.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

visualization_directory = Path('../data/visualizations')
outputs_directory = Path('../data/outputs')
    
def load_metrics():
    '''
    Loads the model performance metrics from the CSV file
    
    Returns:
        pd.DataFrame: The dataframe containing the performance metrics
    '''
    try:
        file_path = outputs_directory / 'model_performance.csv'
        metrics_df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded model performance metrics from {file_path}")
        return metrics_df
    except Exception as e:
        logging.error("Error occurred while loading the model performance metrics", exc_info=True)
        raise

def scatter_regression(X_test, y_test, y_pred): 
    '''
    Creates a scatterplot with a regression line for model performance metrics.
    The line is the predicted values, the scatters are the actual value.
    
    Parameters: 
        X_test: The test features used in the regression model.
        y_test: Test target values.
        y_pred: Predicted target values from the regression model
        
    Returns: 
        A PNG file with the scatter plot and regression line.
    '''