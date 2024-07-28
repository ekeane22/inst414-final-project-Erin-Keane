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

flights_1_df = pd.read_csv("Flights_2022_1.csv")
flights_1_df.head

flights_2_df = pd.read_csv("Flights_2022_2.csv")
flights_2_df.head

flights_3_df = pd.read_csv("Flights_2022_3.csv")
flights_3_df.head

flights_4_df = pd.read_csv("Flights_2022_4.csv")
flights_4_df.head

flights_5_df = pd.read_csv("Flights_2022_5.csv")
flights_5_df.head

flights_6_df = pd.read_csv("Flights_2022_6.csv")
flights_6_df.head

flights_7_df = pd.read_csv("Flights_2022_7.csv")
flights_7_df.head

flights_8_df = pd.read_csv("Flights_2022_8.csv")
flights_8_df.head

flights_9_df = pd.read_csv("Flights_2022_9.csv")
flights_9_df.head

flights_10_df = pd.read_csv("Flights_2022_10.csv")
flights_10_df.head

flights_11_df = pd.read_csv("Flights_2022_11.csv")
flights_11_df.head

flights_12_df = pd.read_csv("Flights_2022_12.csv")
flights_12_df.head

weather_events_df = pd.read_csv("WeatherEvents_Jan2016-Dec2022.csv")
weather_events_df.head()

flights_2022 = [flights_1_df, flights_2_df, flights_3_df, flights_4_df, flights_5_df, flights_6_df, flights_7_df, flights_8_df, flights_9_df, flights_10_df, flights_11_df, flights_12_df]

semi_combined = pd.concat(flights_2022, ignore_index=True)


print (semi_combined)


# load the new combined df to a new dataframe in /data
# use the .to_csv method 

             

