
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

logging.basicConfig(
    filename='transform.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

loaded_directory = Path('../data/loaded')
outputs_directory = Path('../data/outputs')


def read_csv(file_name):
    '''
    Reads the 'wf.csv' file from the loaded directory.

    Returns:
        pd.DataFrame: The dataframe read from the CSV file.
    '''
    try:
        file_path = loaded_directory / 'wf.csv'
        df = pd.read_csv(file_path, low_memory=False)
        logging.info("Successfully read 'wf.csv' from the loaded directory")
        return df
    except Exception as e:
        logging.error("Error occurred while reading 'wf.csv'", exc_info=True)
        raise
    
def preprocessing()
