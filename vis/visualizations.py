import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging 

logger = logging.getLogger(__name__)

outputs_directory = Path('data/outputs')
visualization_directory = Path('data/visualizations')


def read_data_for_eda(file_name='wf_preprocessed.csv'):
    '''
    Reads the file in for EDA analysis on the wf_preprocessed.csv
    
    Returns: 
        DF: the dataframe of the csv file.
    '''
    try: 
        file_path = outputs_directory / file_name
        df = pd.read_csv(file_path)
        logging.info("Successfully read 'wf.csv' from the loaded directory")
        return df
    except Exception as e:
        logging.error("Error occurred while reading 'wf.csv'", exc_info=True)
        raise
def event_type_bar_chart(df): 
    '''
    Bar Cahrt for 'EventType' counts.
    '''
    try: 
        plt.figure(figsize=(12, 8))
        sns.countplot(data=df,
                      x='EventType',
                      palette='viridis')
        plt.title('Count of All Weather Event Types')
        plt.xticks(rotation=45)
        plt.tight_layout
        plt.savefig(visualization_directory / 'eda_event_type_counts.png')
        logging.info("Bar chart for 'EventType' saved successfully")
    except Exception as e: 
        logging.error("Error generating the bar chart.")
        
def weather_delay_boxplot(df):
    '''
    Box plot for 'WeatherDelay' and 'EventType'
    
    '''
    try: 
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='EventType', y='WeatherDelay', palette='viridis')
        plt.title('Box Plot of Weather Delay by Event Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(visualization_directory / 'eda_weather_delay_boxplot.png')
        plt.close()
        logging.info("Box plot for 'WeatherDelay' by 'EventType' saved successfully.")
    except Exception as e:
        logging.error("Error generating box plot")


def main(): 
    df = read_data_for_eda()

    # Create visualizations
    event_type_bar_chart(df)
    weather_delay_boxplot(df)
    
if __name__ == '__main__':
    main()