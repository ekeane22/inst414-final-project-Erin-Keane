import matplotlib.pyplot as plt 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging 

logger = logging.getLogger(__name__)

outputs_directory = Path('data_ek/data/outputs')
visualization_directory = Path('data_ek/data/visualizations')


def read_data_for_eda(file_name='wf_preprocessed.csv'):
    '''
    Reads a CSV file and returns a DataFrame for exploratory data analysis (EDA).

    Args:
        file_name (str): Name of the CSV file to read (default is 'wf_preprocessed.csv').

    Returns:
        df: DataFrame containing the data from the CSV file.
    '''
    try: 
        file_path = outputs_directory / file_name
        df = pd.read_csv(file_path)
        logging.info("Successfully read 'wf.csv' from the loaded directory")
        return df
    except Exception as e:
        logging.error("Error occurred while reading 'wf.csv'", exc_info=True)
        raise
    
def clean_data(df):
    '''
    Removes rows with 0, NaN, or Null values in 'WeatherDelay'.
    
    Args:
        df (DF): The dataframe to clean.
    
    Returns:
        df: The cleaned dataframe.
    '''
    try:
        initial_row_count = df.shape[0]
        df_cleaned = df.dropna(subset=['WeatherDelay'])  
        df_cleaned = df_cleaned[df_cleaned['WeatherDelay'] != 0] 
        cleaned_row_count = df_cleaned.shape[0]
        logging.debug(f"Cleaned data by removing NaN and 0 values in 'WeatherDelay'. Rows before: {initial_row_count}, after: {cleaned_row_count}")
        return df_cleaned
    except Exception as e:
        logging.error("Error occurred while cleaning data", exc_info=True)
        raise   

    
def event_type_bar_chart(df): 
    '''
    Generates and saves a bar chart showing counts of different weather event types.

    Args:
        df (DF): The DataFrame containing the 'EventType' column.

    '''
    try: 
        df_cleaned = clean_data(df)
        plt.figure(figsize=(12, 8))
        sns.countplot(data=df_cleaned,
                      x='EventType',
                      palette='viridis')
        plt.title('Count of All Weather Event Types')
        plt.xticks(rotation=90, ha='center', fontsize=10)
        plt.xlabel('Event Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(visualization_directory / 'eda_event_type_counts.png')
        plt.show()
        logging.info("Bar chart for 'EventType' saved successfully")
    except Exception as e: 
        logging.error("Error generating the bar chart.")
        
def weather_delay_boxplot(df):
    '''
    Generates and saves a box plot showing the distribution of weather delays by event type.

    Args:
        df (DF): The DataFrame containing 'WeatherDelay' and 'EventType' columns.
    '''
    try: 
        df_cleaned = clean_data(df)
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_cleaned, x='EventType', y='WeatherDelay', palette='viridis')
        plt.title('Box Plot of Weather Delay by Event Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(visualization_directory / 'eda_weather_delay_boxplot.png')
        plt.show()
        logging.info("Box plot for 'WeatherDelay' by 'EventType' saved successfully.")
    except Exception as e:
        logging.error("Error generating box plot")


def main(): 
    '''
    Main function to read data, generate, and save visualizations for EDA.
    '''
    df = read_data_for_eda()

    # Create visualizations
    event_type_bar_chart(df)
    weather_delay_boxplot(df)
    
if __name__ == '__main__':
    main()