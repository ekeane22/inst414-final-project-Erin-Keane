# inst414-final-project-Erin-Keane
INST414 Final Project - Airport Project 

# Project Overview 
#### The business problem I am addressing is to identify the relationship between specific weather conditions and their impact on flight delays at Chicago O’Hare International Airport. The two datasets, "Weather Events Data" and "Predicting_Flight_Delays (2022 - US) Raw Data" provide data such as weather events, flight records, and delay/cancellation rates. Through predictive analysis, I am evaluating the relationship between weather conditions and flight delays through a simple linear regression. This provides insights to the Weather Operations Department at Chicago O'Hare International Airport and can improve operations and preparations during inclement weather.


# Set Up Instructions
#### The link for the Data located in the Google Drive: https://drive.google.com/drive/folders/1Uxc7B8nxQeL2OsyVz_CJU0bF-OcaJY_c?usp=sharing

#### 1. Enter this in the terminal: 'python3 -m venv .venv' 
#### 2. Entier this in the terminal: 'source .venv/bin/activate'
#### 3. Clone the repository (using "git clone {repository URL}")
#### 4. Download the data from the Google Drive folder called 'data_ek'. 
#### 5. Unzip the data 
#### 6. Place the Google Drive folder called 'data_ek' into the project folder called 'inst414-final-project-Erin-Keane'.
#### 7. Now, you should have a data directory called 'data_ek/data'
#### 8. Enter this in the terminal: 'cd inst414-final-project-Erin-Keane'
#### 9. Enter this in the ternminal: 'pip install -r requirements.txt'

# Running The Project 
#### 10. To run the main.py, enter "python3 main.py" in the terminal.
#### Note: When main.py is finished running, 5 different visualizations will pop up on your screen. This process may take a minute or two. 

# Code Package Structure 
### inst414-final-project-Erin-Keane

### data_ek/data (From the Google Drive)
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
