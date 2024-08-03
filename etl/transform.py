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


#going to load previous csv, semi_combined

#need to load the weather_events_df

def transform(semi_combined, weather_events_df):
    '''
    Fixes the dataframe and merges the two dataframes together to create combined. 
    '''
    
# wtf what do i merge it on or should i use join, hm not sure... to be discovered 
#im pretty sure I want to combine based on dates   
    combined = pd.merge(semi_combined, weather_events_df, left_on='something', right_on='something else')
    combined = combined.drop("names, axis = 1")
    combined = combined.dropna()
    combined.drop_duplicates()
    combined.reset_index(drop=True)
    
    
    return combined 

def rename(combined, new):
    #wtf do i have to rename... 
    # WAIT CONCAT THOSE TWO LOCATION COLUMNS FOR WEATHER 
    # Ill probably join on those 
    '''
    allows me to create new names for the columns. 
    I will have a dictionary to do this. 
    '''
    new = {dictionary}
    
    combined = combined.rename(columns=new)
    return combined  


def drop(combined): 
    '''
    Drops any columns I dont want in my cleaned df. 
    Args: 
        Df: the combined dataframe of weather and flight data 
    Returns: 
        df: the updated combined dataframe. 
    '''
    remove = ['columnA', 'columnB']
    combined = combined.drop(columns=remove)
    
    return combined 


#In row cancellation code == drop A, C, and D 
#those are not weather delays 


#save combined in new csv 
    #print combined 
