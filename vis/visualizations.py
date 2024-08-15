'''
Bar Charts, Line Graph, Histograms, Scatte plots.

Can be exploratory and rough. 

I saw a picture of a scatter_geo based on clusers, might try. 

I think im going to revise and build different dataframes 
based on the graph I want to make... Maybe.  
'''

import plotly 
import bokeh 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
from pathlib import Path
import logging 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

logging.basicConfig(
    filename='vis.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

visualization_directory = Path('../data/visualizations')
outputs_directory = Path('../data/outputs')
    


