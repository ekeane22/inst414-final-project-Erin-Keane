import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


'''
Downloads the data dictionary csvs
'''



flight_delays = "Data Dictionary_Predicting Flight Delays 2022.csv"
data = pd.read_csv(flight_delays, encoding='utf-8')
print(data.head())
# im not sure why weather wont fully import the csv in correct format. 
weather = "Data Dictionary_US Weather Events (2016 - 2022).csv"
data2 = pd.read_csv(weather, encoding='utf-8')
print(data2.head())



#Predicting Flight Delays 	
#Attribute	Reference 
#Cancelled Flight 	1 = Yes 
#Diverted Flight 	1 = Yes 

