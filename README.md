# inst414-final-project-Erin-Keane
INST414 Final Project - Airport Project 

# Project Overview 
#### The business problem I am addressing is to identify the relationship between specific weather conditions and their impact on flight delays at Chicago O’Hare International Airport. The two datasets, "Weather Events Data" and "Predicting_Flight_Delays (2022 - US) Raw Data" provide data such as weather events, flight records, and delay/cancellation rates. Through predictive analysis, I am evaluating the relationship between weather conditions and flight delays through a simple linear regression. This provides insights to the Weather Operations Department at Chicago O'Hare International Airport and can improve operations and preparations during inclement weather.


# Set Up Instructions/ Install Dependencies 
### To set up Venv in VS code, enter these commands in the terminal: 
#### 1. python -m venv .venv 
#### 2. source .venv/bin/activate
#### 3. Clone the repository (using "git clone repository URL")
#### 4. Download the data from the Google Drive folder called 'final-project-data-EK'. 
#### 5. Unzip the data 
#### 6. Place the Google Drive folder called 'final-project-data-EK' into the VS Code directory called 'data/'
#### 7. Now, you should have a data directory called 'data/final_project_data_EK
#### 8. In your terminal, enter 'cd inst414-final-project-Erin-Keane'
#### 9. pip install -r requirements.txt 

# Running The Project 
#### To run the main.py, enter "python3 main.py" in the terminal.

# Code Package Structure 
### inst414-final-project-Erin-Keane

### data/final-project-data-EK
#### original_data/
#### extracted/
##### transformed/
##### loaded/
##### outputs/ 
##### visualizations/
##### reference_tables/

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
