
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

logging.basicConfig(
    filename='etl.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    try:
        loaded_directory.mkdir(parents=True, exist_ok=True)
        src_file = transformed_directory / 'wf.csv'
        dest_file = loaded_directory / 'wf.csv'

        shutil.copy(src_file, dest_file)
        logging.info(f"Successfully copied 'wf.csv' from {transformed_directory} to {loaded_directory}")
        
        df_wf = pd.read_csv(dest_file, low_memory=False)
        logging.info("Loaded 'wf.csv' into dataframe")
        
        return df_wf
    except FileNotFoundError as e: 
        logging.error("The file was not found when copying", exc_info=True)
        raise 
    except Exception as e: 
        logging.error("Error copying the file or loading the dataframe")
        raise


def main(): 
    df_wf = copy_wf(transformed_directory, loaded_directory)
    print(df_wf.head())
    
if __name__ == "__main__": 
    main()