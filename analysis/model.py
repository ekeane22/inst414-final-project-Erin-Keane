'''
MODEL IS FIRST, THEN EVALUATE 

Model: 
    multiple linear Regression 
    
    
reads csv from loaded directory. 
at the end it will save the new csv to the outputs directory. 

'''


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

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
    wf = pd.read_csv(loaded_directory / 'wf.csv')
   
    return wf 

def preprocess(wf): 
    
    
#handle missing values 
#cancelled 
#cancellation code b = weather 
#carrierdelay 
#weather delay 
#NASdelay 
#security delay 
#late aircraft delay 




