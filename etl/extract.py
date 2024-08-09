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
import mylib 
   
data_directory = Path('../data/original data')
extracted_directory = Path('../data/extracted')

flight1_df = pd.read_csv(data_directory / "Flight1.csv")
flight2_df = pd.read_csv(data_directory / "Flight2.csv")
flight3_df = pd.read_csv(data_directory / "Flight3.csv")
flight4_df = pd.read_csv(data_directory / "Flight4.csv")
flight5_df = pd.read_csv(data_directory / "Flight5.csv")
flight6_df = pd.read_csv(data_directory / "Flight6.csv")
flight7_df = pd.read_csv(data_directory / "Flight7.csv")
flight8_df = pd.read_csv(data_directory / "Flight8.csv")
flight9_df = pd.read_csv(data_directory / "Flight9.csv")
flight10_df = pd.read_csv(data_directory / "Flight10.csv")
flight11_df = pd.read_csv(data_directory / "Flight11.csv")
flight12_df = pd.read_csv(data_directory / "Flight12.csv")

lake_df = pd.read_csv(data_directory / "storm_data_lake.csv")
kendall_df = pd.read_csv(data_directory / "storm_data_kendall.csv")
grundy_df = pd.read_csv(data_directory / "storm_data_grundy.csv")
will_df = pd.read_csv(data_directory / "storm_data_will.csv")
kankakee_df = pd.read_csv(data_directory / "storm_data_kankakee.csv")
cook_df = pd.read_csv(data_directory / "storm_data_cook.csv")
mchenry_df = pd.read_csv(data_directory / "storm_data_mchenry.csv")
kane_df = pd.read_csv(data_directory / "storm_data_kane.csv")
dupage_df = pd.read_csv(data_directory / "storm_data_dupage.csv")
dekalb_df = pd.read_csv(data_directory / "storm_data_dekalb.csv")


f2022 = [flight1_df, flight2_df, flight3_df, flight4_df, flight5_df, flight6_df, flight7_df, flight8_df, flight9_df, flight10_df, flight11_df, flight12_df]
flight22 = pd.concat(f2022, ignore_index=True)


county = [lake_df, kendall_df, grundy_df, will_df, kankakee_df, cook_df, mchenry_df, kane_df, dupage_df, dekalb_df]
county22 = pd.concat(county, ignore_index=True)

flight22.to_csv(extracted_directory / 'flight22.csv', index=False)
county22.to_csv(extracted_directory / 'county22.csv', index=False)
