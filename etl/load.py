
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import shutil 
import logging 

logger = logging.getLogger(__name__)

transformed_directory = Path('data/final_project_data_EK/transformed')
loaded_directory = Path('data/final_project_data_EK/loaded')
    

def copy_wf(transformed_directory, loaded_directory): 
    '''
    Copies the 'wf.csv' file from the transformed directory to the loaded directory and loads it into a DataFrame.

    Args:
        transformed_directory (Path): Path to the source directory containing the 'wf.csv' file.
        loaded_directory (Path): Path to the destination directory where the 'wf.csv' file will be copied.

    Returns:
        DF: DataFrame loaded from the 'wf.csv' file in the destination directory.

    Raises:
        FileNotFoundError: If the 'wf.csv' file is not found in the source directory.
        Exception: For other errors that occur during copying or loading the file.
    
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
    '''
    Main function to execute the copying and loading of 'wf.csv' file from the transformed directory to the loaded directory.
    '''
    df_wf = copy_wf(transformed_directory, loaded_directory)
    print(df_wf.head())
    
if __name__ == "__main__": 
    main()