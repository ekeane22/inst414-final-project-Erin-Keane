
# im not sure what im supposed to put here 
#load needs to have info bc it goes from one database into another data system
# data engineers are running security checks and logs and quality assurance checks 
# makes sure data is working right and it has integrity and it loads correctly 


'''
technically, in this function, going to pull from transformed directory to here 
then, move it to the loaded folder in the data directory


'''

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import shutil 
import logging 

transformed_directory = Path('../data/transformed')
loaded_directory = Path('../data/loaded')
    

def copy_wf(transformed_directory, loaded_directory): 
    '''
    Copies the wf.csv file from the transformed directory into the loaded directory. 
    
    Args: 
        transformed_directory (Path): source path 
        loaded_directory (Path): destination path 
    
    Returns: 
        pd.DataFrame: dataframe that is moved to destination path. 
    
    '''
    new_wf = pd.read_csv(transformed_directory / 'wf.csv', low_memory=False)
    new_wf.to_csv(loaded_directory / 'wf.csv', index=False)
    
    return new_wf 


def main(): 
    df_wf = copy_wf(transformed_directory, loaded_directory)
    print(df_wf.head())
    
if __name__ == "__main__": 
    main()