'''
Extract.py is designed to extract code from the data sources. 
In this case I will be downloading and reading .csv files. 
The extracted data is stored in extracted.py. 

The two different datasets/ flat files that I need to download and read: 
Kaggle - US Weather Events (2016 - 2022) 
    https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events 
Kaggle - Predicting_Flight_Delays (2022 - US) Raw Data
    https://www.kaggle.com/datasets/omerkrbck/1-raw-data-predicting-flight-delays?select=Flights_2022_5.csv 


Returns: 
    df: flight 2022 concatonated dataframe with the flight data from the 12 months. 
'''

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# These csv's wont load, I need to pre-clean them and redownload. 
# they are millions of rows 

FFlight1_df = pd.read_csv("FFlight_1.csv")
FFlight1_df.head

FFlight2_df = pd.read_csv("FFlight_2.csv")
FFlight2_df.head

FFlight3_df = pd.read_csv("FFlight_3.csv")
FFlight3_df.head

FFlight4_df = pd.read_csv("FFlight_4.csv")
FFlight4_df.head

FFlight5_df = pd.read_csv("FFlight_5.csv")
FFlight5_df.head

FFlight6_df = pd.read_csv("FFlight_6.csv")
FFlight6_df.head

FFlight7_df = pd.read_csv("FFlight_7.csv")
FFlight7_df.head

FFlight8_df = pd.read_csv("FFlight_8.csv")
FFlight8_df.head

FFlight9_df = pd.read_csv("FFlight_9.csv")
FFlight9_df.head

FFlight10_df = pd.read_csv("FFlight_10.csv")
FFlight10_df.head

FFlight11_df = pd.read_csv("FFlight_1.csv")
FFlight11_df.head

FFlight12_df = pd.read_csv("FFlight_12.csv")
FFlight12_df.head

weather_events_df = pd.read_csv("WeatherEvents_Jan2016-Dec2022.csv")
weather_events_df.head()

flights_2022 = [flights_1_df, flights_2_df, flights_3_df, flights_4_df, flights_5_df, flights_6_df, flights_7_df, flights_8_df, flights_9_df, flights_10_df, flights_11_df, flights_12_df]

semi_combined = pd.concat(flights_2022, ignore_index=True)


print (semi_combined)


# load the new combined df to a new dataframe in /data
# use the .to_csv method 

             

