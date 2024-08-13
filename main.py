import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


data_directory = Path('../data/original data')
extracted_directory = Path('../data/extracted')
transformed_directory = Path('../data/transformed')
loaded_directory = Path('../data/loaded')
outputs_directory = Path('../data/outputs')
visualizations_directory = Path('../data/visualizations')

from extract import load_flight_data, load_storm_data, save_data

def main():
# Load data
    flight_data = load_flight_data()
    storm_data = load_storm_data()
    
    # Save data
    save_data(flight_data, storm_data)