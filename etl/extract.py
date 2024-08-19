'''
Extracts multiple csv files from the "original data" folder of the data directory, reads the csv files, concatonates them,
and saves the new concatenated csv's in the "extracted" folder of the data directory. 

The Flight csv flat files were found on Kaggle, called "Predicting_Flight_Delays(2022 - US) Raw_Data."
This data is originates from the Department of Transportatoins Bureau of Transportation Statistics. 
This data was likely collected by wab scraping skills before it was transformed and posted as csv files on Kaggle.
    https://www.kaggle.com/datasets/omerkrbck/1-raw-data-predicting-flight-delays?select=Flights_2022_5.csv 

The Storm csv flat files are from the National Oceanic and Atmospheric Administration's (NOAA) and their National Centers for Environmental Information. 
his dataset was collected from several different satellites, the National Weather Service, county, state, and federal emergency management, 
local law enforcement, insurance requests, and the general public. The National Centers for Environmental Information takes this information, 
transforms it, and releases the data in CSV format 75 days after the end of the month. 
https://www.ncdc.noaa.gov/stormevents/choosedates.jsp?statefips=17%2CILLINOIS 

Both Datasets are filtered to focus on flights and weather event data for Chicago O'Hare International Airport

Returns: 
    flight22 (df): concatenated flight data.
    county22 (df): concatenated county weather event data. 

'''

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging 

logger = logging.getLogger(__name__)
   
data_directory = Path('data_ek/data/original_data')
extracted_directory = Path('data_ek/data/extracted')

def load_flight_data(): 
    '''
    Loads and concatenates CSV files containing flight data.
    
    The function reads 12 CSV files that corresponds the monthly flight data for the year 2022. 
    It concatenates all 12 CSVs into a single DataFrame, and returns it. 
    
    Returns: 
        flight_df (DF): Concatenated flight data.
    
    '''
    try: 
        flight_files = [f"Flight{i}.csv" for i in range(1, 13)]
        logger.debug(f"Data directory: {data_directory}")
        logger.debug(f"Loading the flight data from the directory: {flight_files}")
        flight_df = [pd.read_csv(data_directory / file) for file in flight_files]
        result = pd.concat(flight_df, ignore_index=True)
        logger.info(f"Successfully loaded the flight data")
        return result
    except Exception as e: 
        logger.error("Error loading the flight data", exc_info=True)
        raise 

def load_storm_data():
    '''
    Loads and concatenates CSV files constaining storm data. 
    
    The function reads multiple CSV files corresponding to weather event data from the 10
    counties that surround Chicago O'Hare International Airport within 50 miles. 
    
    Concatenates the 10 CSV files into a single DataFrame, and Returns it. 
    
    Returns: 
        storm_df (DF): Concatenated storm data.
    
    '''
    
    try: 
        storm_files = [
            "storm_data_lake.csv", "storm_data_kendall.csv", "storm_data_grundy.csv", 
            "storm_data_will.csv", "storm_data_kankakee.csv", "storm_data_cook.csv", 
            "storm_data_mchenry.csv", "storm_data_kane.csv", "storm_data_dupage.csv", 
            "storm_data_dekalb.csv"
        ]
        logging.debug(f"Loading the storm data from the directory: {storm_files}")
        storm_df = [pd.read_csv(data_directory / file) for file in storm_files]
        result = pd.concat(storm_df, ignore_index=True)
        logging.info(f"Successfully loaded storm data")
        return result
    except Exception as e: 
        logging.error("Error loading the storm data", exc_info=True)
        raise

def save(flight_df, storm_df): 
    '''
    Saves the concatenated flight and storm data to CSV files. 
    
    Args: 
        flight_df (DF): The DataFrame containing concatenated flight data.
        storm_df (DF): The DataFrame containing concatenated storm data.
        
    Returns: 
        Exception if an error occurs while saving the data.

    '''
    try: 
        flight_df.to_csv(extracted_directory / 'flight22.csv', index=False)
        storm_df.to_csv(extracted_directory / 'county22.csv', index=False)
        logging.info("Data was successfully saved to CSV files")
    except Exception as e: 
        logging.error("Error saving data", exc_info=True)
        raise
    
def process_data(): 
    '''
    Processes the flight and storm data by loading, concatenating, and saving.

    The function loads flight and storm data, concatenates the data from multiple CSV files,
    and saves the result to the 'extracted' directory.
    
    Raises:
        Exception if an error occurs during data processing.
    '''
    try: 
        
        # Ensure the extracted directory exists
        extracted_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {extracted_directory}")  
        
        flight_data = load_flight_data()
        storm_data = load_storm_data()
        save(flight_data, storm_data)
    except Exception as e: 
        logging.error("Error processing data", exc_info=True)
        raise 
    
if __name__=="__main__": 
    process_data()