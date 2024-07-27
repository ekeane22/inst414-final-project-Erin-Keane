'''
Extract.py is designed to extract code from the data soruces. 
In this case I will be downloading and reading .csv files. 
The extracted data is stored in etl /data. 

The two different datasets/ flat files that I need to download and read: 
Kaggle - US Weather Events (2016 - 2022) 
    https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events 
Kaggle - Predicting_Flight_Delays (2022 - US) Raw Data
    https://www.kaggle.com/datasets/omerkrbck/1-raw-data-predicting-flight-delays?select=Flights_2022_5.csv 

'''

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

flights_1_df = pd.read_csv("Flights_2022_1.csv")
print(flights_1_df.head())
