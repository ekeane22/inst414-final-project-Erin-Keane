import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging 
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    filename='evaluate.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

loaded_directory = Path('../data/loaded')
outputs_directory = Path('../data/outputs')

def read_csv(file_name='wf.csv'):
    '''
    Reads the 'wf.csv' file from the loaded directory.

    Returns:
        pd.DataFrame: The dataframe read from the CSV file.
    '''
    try:
        file_path = loaded_directory / file_name
        df = pd.read_csv(file_path, low_memory=False)
        logging.info("Successfully read 'wf.csv' from the loaded directory")
        return df
    except Exception as e:
        logging.error("Error occurred while reading 'wf.csv'", exc_info=True)
        raise

def preprocessing1(df): 
    columns_to_drop = [
        'Quarter', 'OriginStateName', 'OriginState', 'DestState', 'DestStateName', 
        'CarrierDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 
        'AbsoluteRowNumber', 'BeginLocation', 'TorFScale', 'DeathsDirect', 
        'InjuriesDirect', 'DamagePropertyNum', 'DamageCropsNum', 'State', 
        'CZTimezome', 'MagnitudeType', 'EpisodeID', 'CZ_TYPE', 'CZFips', 'WFO', 
        'InjuriesIndirect', 'DeathsIndirect', 'Source', 'FloodCause', 'TorLength', 
        'TorWidth', 'BeginRange', 'BeginAzimuth', 'EndRange', 'EndAzimuth', 
        'EndLocation', 'BeginLat', 'BEGIN_LON', 'EndLat', 'EventNarrative', 
        'EpisodeNarrative', 'END_LON', 'Month', 'DayofMonth', 'DayOfWeek'
    ] 
    try: 
        df.drop(columns=columns_to_drop, inplace=True)
        logging.info("The specified columns were dropped from the dataframe")
       
        # Removing rows where the cancellation code is 'A', 'C', or 'D'
        initial_row_count = df.shape[0]
        df = df[~df['CancellationCode'].isin(['A', 'C', 'D'])]
        filtered_row_count = df.shape[0]
        logging.debug(f"Filtered rows where CancellationCode is 'a', 'c', or 'd'. Rows before: {initial_row_count}, after: {filtered_row_count}")

        # Remove rows where the weather event is 'Drought' or 'Rip Current'
        initial_row_count = df.shape[0]
        df = df[~df['EventType'].isin(['Drought', 'Rip Current'])]
        filtered_row_count = df.shape[0]
        logging.debug(f"Filtered rows where EventType is 'Drought' or 'Rip Current'. Rows before: {initial_row_count}, after: {filtered_row_count}")
        
        # Moving the 'ID' column to the beginning
        id_column = df.pop('ID')
        df.insert(0, 'ID', id_column)
        
        # Resetting the 'ID' column to start at 1
        df['ID'] = range(1, len(df) + 1)
        logging.debug("Moved 'ID' column to the beginning of the dataframe and reset 'ID' to start at 1")

        logging.info("Successfully completed preprocessing1 steps")
        return df
    except Exception as e:
        logging.error("Error occurred during preprocessing1", exc_info=True)
        raise

def preprocessing2(df):
    '''
    Creates interaction terms for feature engineering and applies one-hot encoding.

    Args:
        df (pd.DataFrame): The dataframe to preprocess.

    Returns:
        pd.DataFrame: The dataframe with new interaction columns and one-hot encoded features.
    '''
    try:
        # Replace NaN values in 'CancellationCode' and 'WeatherDelay'
        df['CancellationCode'] = df['CancellationCode'].fillna(0)
        df['WeatherDelay'] = df['WeatherDelay'].fillna(0)
        
        # Creating interaction terms
        df['EventType_Cancellation'] = df['EventType'].astype(str) + '_' + df['Cancelled'].astype(str)
        df['EventType_CancellationCode'] = df['EventType'].astype(str) + '_' + df['CancellationCode'].astype(str)
        df['EventType_WeatherDelay'] = df['EventType'].astype(str) + '_' + df['WeatherDelay'].astype(str)
        
        # List of columns to one-hot encode
        categorical_features = ['EventType', 'EventType_Cancellation', 'EventType_WeatherDelay']

        # Verify that the columns exist
        missing_cols = [col for col in categorical_features if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns missing from DataFrame: {missing_cols}")

        # Convert columns to strings
        df[categorical_features] = df[categorical_features].astype(str)

        # One-hot encoding
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_features = encoder.fit_transform(df[categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        one_hot_encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

        # Add the 'ID' column to the one-hot encoded dataframe
        one_hot_encoded_df['ID'] = df['ID']
        
        logging.debug("Applied one-hot encoding to the features")
        logging.info("Successfully completed preprocessing2 steps with one-hot encoding")
        return one_hot_encoded_df
    except Exception as e:
        logging.error("Error occurred during preprocessing2", exc_info=True)
        raise
def main():
    try:
        # Read the data
        df_wf = read_csv()
        print("Data read successfully")
        print(df_wf.head())  # Check the initial dataframe

        # Apply preprocessing1
        df_wf = preprocessing1(df_wf)
        print("Completed preprocessing1")
        print(df_wf.head())  # Check the dataframe after preprocessing1
        
        # Save the intermediate dataframe after preprocessing1
        output_file_path_intermediate = outputs_directory / 'wf_preprocessed.csv'
        df_wf.to_csv(output_file_path_intermediate, index=False)
        logging.info(f"Successfully saved the dataframe after preprocessing1 to {output_file_path_intermediate}")
        print(f"Data after preprocessing1 saved to {output_file_path_intermediate}")

        # Apply preprocessing2 with one-hot encoding
        df_wf_encoded = preprocessing2(df_wf)
        print("Completed preprocessing2")
        print(df_wf_encoded.head())  # Check the dataframe after preprocessing2

        # Save the updated dataframe to the outputs directory
        output_file_path_encoded = outputs_directory / 'wf_preprocessed_encoded.csv'
        df_wf_encoded.to_csv(output_file_path_encoded, index=False)
        logging.info(f"Successfully saved the preprocessed and encoded dataframe to {output_file_path_encoded}")
        print(f"Data saved to {output_file_path_encoded}")

    except Exception as e:
        logging.error("Error occurred in the main function", exc_info=True)
        raise

if __name__ == "__main__":
    main()


