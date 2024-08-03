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


'''
These .CSV viles are from Kaggle, however, they have different original sources. 
ENTER MORE INFO HERE 

'''     

FFlight1_df = pd.read_csv("FFlight_1.csv")
FFlight2_df = pd.read_csv("FFlight_2.csv")
FFlight3_df = pd.read_csv("FFlight_3.csv")
FFlight4_df = pd.read_csv("FFlight_4.csv")
FFlight5_df = pd.read_csv("FFlight_5.csv")
FFlight6_df = pd.read_csv("FFlight_6.csv")
FFlight7_df = pd.read_csv("FFlight_7.csv")
FFlight8_df = pd.read_csv("FFlight_8.csv")
FFlight9_df = pd.read_csv("FFlight_9.csv")
FFlight10_df = pd.read_csv("FFlight_10.csv")
FFlight11_df = pd.read_csv("FFlight_11.csv")
FFlight12_df = pd.read_csv("FFlight_12.csv")
weather_df = pd.read_csv("WeatherEvents2022.csv")

print(FFlight1_df.head())
print(FFlight2_df.head())
print(FFlight3_df.head())
print(FFlight4_df.head())
print(FFlight5_df.head())
print(FFlight6_df.head())
print(FFlight7_df.head())
print(FFlight8_df.head())
print(FFlight9_df.head())
print(FFlight10_df.head())
print(FFlight11_df.head())
print(FFlight12_df.head())
print(weather_df.head())

flight_2022 = [FFlight1_df, FFlight2_df, FFlight3_df, FFlight4_df, FFlight5_df, FFlight6_df, FFlight7_df, FFlight8_df, FFlight9_df, FFlight10_df, FFlight11_df, FFlight12_df]
semi_combined = pd.concat(flight_2022, ignore_index=True)

print(semi_combined)

semi_combined.to_csv('data/combined_flights_2022.csv', index=False)

#HELP WTF WHY CANT I GET IT TO WORK 
