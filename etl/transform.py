'''
Write code to pull the data in from the file that the csv was stored in. 
This is /data under etl. 
This involves making decisions and handling missing values, duplicates, consistent attribute naming, 
robust unique IDs, tidying and normalizing data, joining and merging datasets. Remember that the goal 
is to maintain the integrity of the data and transform it into analytical data.

In this .py, I need to make edits to the csv's, one of the .csv's includes lots of rows and columns I dont need. 

Returns: 
    df: The fully combined dataframe from the weather and the flights. 
'''


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging 

logger = logging.getLogger(__name__)

extracted_directory = Path('data/extracted')
transformed_directory = Path('data/transformed')

def load_from_extracted(extracted_directory): 
    '''
    Loads the two csv dataframes from the extract folder from the data directory. 
    
    Args: 
        extracted_directory (Path): Path to the folder containing the dataframes from extracted.py.
        
    Returns: 
        tuple: Contains the county22 and flight22 DataFrames. 
    '''
    try: 
        county22 = pd.read_csv(extracted_directory / 'county22.csv')
        flight22 = pd.read_csv(extracted_directory / 'flight22.csv')
        logging.info("Successfully loaded data from extracted directory")
        return county22, flight22
    except FileNotFoundError as e: 
        logging.error("The files were not found while loading data", exc_info=True)
        raise 
    except Exception as e: 
        logging.error("Error loading data from the extracted directory", exc_info=True)
        raise 
def add_id_column(flight22): 
    '''
    Adds a unique ID column for flight22 DF. 
    
    Args: 
        flight22 (Df): Contains flight information for Chicago O'Hare Airport.
        
    Returns: 
        DF: Updated flight22 DF with new "ID" Column.
    
    '''
    flight22['ID'] = range(1, len(flight22) + 1)
    logging.debug("Added a unique ID column to flight22 dataframe")
    return flight22

def fix_county22(county22):
    '''
    Moves the Absoulte row number from the last column to the first columnin the county22 DF. 
    
    Args: 
        county22 (DF): Contains county weather events that surround Chicago O'Hare Airport.
        
    Returns: 
        df: new county22 df with absoulte row number as the first column and not the last and the update time column. 
    
    '''
    try: 
        columns = ["ABSOLUTE_ROWNUMBER"] + [col for col in county22.columns if col != "ABSOLUTE_ROWNUMBER"]
        county22 = county22[columns]
        logging.debug("Successfully reordered columns in the county22 dataframe")
        return county22
    except Exception as e: 
        logging.error("Error fixing the countu22 dataframe", exc_info=True)
        raise 


def rename1(county22):
    '''
    Renames the columns in the county22 dataframe to match flight22 dataframe. 
    This promotes data integrity and data tidiness. 
    
    Args: 
        county22 (DF): Contains county weather events that surround Chicago O'Hare Airport.
        
    Returns: 
        count22 (DF): updated Dataframe with consistent column names
        
        '''
    try: 
        county22 = county22.rename(columns={
            'EVENT_ID': 'EventID', 
            'CZ_NAME_STR': 'CZNameStr',
            'BEGIN_LOCATION': 'BeginLocation',
            'BEGIN_DATE': 'BeginDate',
            'BEGIN_TIME': 'BeginTime',
            'EVENT_TYPE': 'EventType',
            'MAGNITUDE': 'Magnitude',
            'TOR_F_SCALE': 'TorFScale',
            'DEATHS_DIRECT': 'DeathsDirect',
            'INJURIES_DIRECT': 'InjuriesDirect',
            'DAMAGE_PROPERTY_NUM': 'DamagePropertyNum',
            'DAMAGE_CROPS_NUM': 'DamageCropsNum',
            'STATE_ABBR': 'State', 
            'CZ_TIMEZONE': 'CZTimezome',
            'MAGNITUDE_TYPE': 'MagnitudeType',
            'EPISODE_ID': 'EpisodeID', 
            'CZTYPE': 'CZType', 
            'CZ_FIPS': 'CZFips', 
            'INJURIES_INDIRECT': 'InjuriesIndirect', 
            'DEATHS_INDIRECT': 'DeathsIndirect', 
            'SOURCE': 'Source',
            'FLOOD_CAUSE': 'FloodCause', 
            'TOR_LENGTH': 'TorLength', 
            'TOR_WIDTH': 'TorWidth', 
            'BEGIN_RANGE': 'BeginRange', 
            'BEGIN_AZIMUTH': 'BeginAzimuth', 
            'END_RANGE': 'EndRange', 
            'END_AZIMUTH': 'EndAzimuth', 
            'END_LOCATION': 'EndLocation',
            'END_DATE': 'EndDate', 
            'END_TIME': 'EndTime',
            'BEGIN_LAT': 'BeginLat', 
            'BEGIN_LOT': 'BeginLot', 
            'END_LAT': 'EndLat', 
            'END_LOT': 'EndLot', 
            'EVENT_NARRATIVE': 'EventNarrative', 
            'EPISODE_NARRATIVE': 'EpisodeNarrative', 
            'ABSOLUTE_ROWNUMBER': 'AbsoluteRowNumber'
        })
        logging.debug("Renamed the columns in county22 dataframe")
        return county22
    except Exception as e: 
        logging.error('Error renaming columns in county22 dataframe', exc_info=True)
        raise

def filter_flights(flight22): 
    '''
    Filters the flight22 dataframe for flights originating from "ORD" and where OriginCityName is "Chicago, IL"
    
    Args: 
        flight22 (DF): Contains flight information for Chicago O'Hare Airport.
        
    Returns: 
        Df: Updated dataframe. 
    '''
    try:
        filtered_flights = flight22[(flight22['Origin'] == 'ORD') & (flight22['OriginCityName'] == 'Chicago, IL')]
        logging.debug('Filtered the flights dataframe for "ORD" and "Chicago, IL".')
        return filtered_flights
    except Exception as e:
        logging.error('Error filtering flights dataframe', exc_info=True)
        raise

def merge(flight22, county22): 
    '''
    Merges flight22 and county22 dataframes on BeginDate and FlightDate
    
    Args: 
        county22 (DF): Contains county weather events that surround Chicago O'Hare Airport.
        flight22 (DF): Contains flight information for Chicago O'Hare Airport.
        
    Returns: 
        wf (DF): New merged dataframe with all columns from both dat asets.
    
    '''
    try:
        fflights22 = flight22.copy()
        fflights22['FlightDate'] = pd.to_datetime(fflights22['FlightDate'])
        county22['BeginDate'] = pd.to_datetime(county22['BeginDate'])
        merged_df = pd.merge(fflights22, county22, left_on='FlightDate', right_on='BeginDate', how='outer')
        merged_df.to_csv(transformed_directory / 'wf.csv', index=False)
        logging.info('Successfully merged the dataframes and saved to wf.csv.')
        return merged_df
    except Exception as e:
        logging.error('Error merging dataframes', exc_info=True)
        raise

def main(): 
    county22, flight22 = load_from_extracted(extracted_directory)
    flight22 = add_id_column(flight22)
    county22 = fix_county22(county22)
    county22 = rename1(county22)
    
    filtered_flights = filter_flights(flight22)
    
    merged_df = merge(filtered_flights, county22)
    
    print(merged_df.head())

if __name__ == "__main__":
    main()