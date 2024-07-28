# inst414-final-project-Erin-Keane
INST414 Final Project - Airport Project 

# Project Overview 
## My business problem involves understanding how weather and seasonal conditions correlate with flight delays and cancellation rates at different U.S. airports.The two datasets, "U.S. Weather Events (2016-2022)" and "Predicting_Flight_Delays (2022 - US) Raw Data" provide data such as weather conditions, flight records, and delay/cancellation rates. Through descriptive analysis, including measures of frequency, central tendency, and dispersion, I aim to highlight the weather conditions and seasons associated with the highest rates of delays and cancellations. This will help airports and airlines better anticipate and manage operations during inclement weather.

# Set Up Instructions 
### To set up Venv in VS code: 
#### 1. python -m venv .venv 
#### 2. source .venc/bin/activate

## Install dependencies 
### pip install -r requirements.txt
### pip isntall pandas 
### pip install matplotlib 
### pip install seaborn 
### pip install numpy 
### pip install plotly 
### pip install brokeh 

# Running The Project 
### To run the main.py, you will enter "python3 main.py" in the terminal. 

# Code Package Structure 
## inst414-final-project-Erin-Keane

## data/
### extracted.py
##### Flights_2022_1.csv
##### Flights_2022_2.csv
##### ...
##### Flights_2022_12.csv
##### WeatherEvents_Jan2016-Dec2022csv
### outputs.py 
### processed.py
### reference-tables.py

## etl/
### extract.py
### transform.py 
### load.py 

## analysis/
### evaluate.py 
### model.py 
### exploration_vis.py 
### aggregation_descriptive_stat.py

## vis/ 
### visualizations.py

## .gitignore
## main.py
## README.md
## requirements.txt

