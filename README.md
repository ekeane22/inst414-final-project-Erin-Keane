# inst414-final-project-Erin-Keane
INST414 Final Project - Airport Project 

# Project Overview 
#### The business problem I am addressing is to identify the relationship between specific weather conditions and their impact on flight delays at Chicago O’Hare International Airport. The two datasets, "Weather Events Data" and "Predicting_Flight_Delays (2022 - US) Raw Data" provide data such as weather events, flight records, and delay/cancellation rates. Through predictive analysis, I am evaluating the relationship between weather conditions and flight delays through a simple linear regression. This provides insights to the Weather Operations Department at Chicago O'Hare International Airport and can improve operations and preperations during inclement weather.


# Set Up Instructions/ Install Dependencies 
### To set up Venv in VS code, enter these commands in the terminal: 
#### 1. python -m venv .venv 
#### 2. source .venv/bin/activate
#### 3. pip install -r requirements.txt

# Running The Project 
#### To run the main.py, enter "python3 main.py" in the terminal.

# Code Package Structure 
### inst414-final-project-Erin-Keane

### data/
#### original data/
#### extracted/
##### transformed/
##### loaded/
##### outputs/ 
##### visualizations/
##### reference tables/

### etl/
#### extract.py
#### transform.py 
#### load.py 

### analysis/
#### evaluate.py 

### vis/ 
#### visualizations.py

### .gitignore
### main.py
### README.md
### requirements.txt