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

# this is rough 
def load_from_extracted(extracted_directory): 
    '''
    Loads the two csv dataframes from the extract folder from the data directory. 
    
    Args: 
        extracted_directory (Path): Path to the folder containing the dataframes from extracted.py.
        
    Returns: 
        tuple: Contains the county22 and flight22 DataFrames. 
    '''
    flight22 = pd.read_csv(extracted_directory / 'flight22.csv', index=False)
    county22 = pd.read_csv(extracted_directory / 'county22.csv', index=False)
    return county22, flight22

def add_id_column(flight22): 
    '''
    Adds a unique ID column for flight22 DF. 
    
    Args: 
        flight22 (Df): Contains flight information for Chicago O'Hare Airport.
        
    Returns: 
        DF: Updated flight22 DF with new "ID" Column.
    
    '''
    flight22['ID'] = range(1, len(flight22) + 1)
    return flight22

def military_time(time_str): 
    '''
    Converts the columns to military time and format "HH:MM".
    
    Args: 
        time_str (str): The time string to convert 
        
    Returns: 
        str: The time in "HH;MM military time format. 
    
    '''
    if pd.isna(time_str): 
        return None 
    
    time_str = str(int(time_str)).zfill(4)
    return f'{time_str[:2]}:{time_str[2:]}'

def fix_flight22(flight22):
    '''
    Adds an ID column to flight22 DF and converts the DepTime column to military time. 
    
    Args: 
        flight22 (DF): Contains flight information for Chicago O'Hare Airport.
        
    Returns: 
        Df: Updated dataframe. 
    '''
    flight22 = add_id_column(flight22)
    flight22['DepTime'] = flight22['DepTime'].apply(military_time)
    return flight22


def fix_county22(county22):
    '''
    Converts BEGIN_TIME column to military time.
    Moves the Absoulte row number from the last column to the first columnin the county22 DF. 
    
    Args: 
        county22 (DF): Contains county weather events that surround Chicago O'Hare Airport.
        
    Returns: 
        df: new county22 df with absoulte row number as the first column and not the last and the update time column. 
    
    '''
    county22['BEGIN_TIME'] = county22['BEGIN_TIME'].apply(military_time)
    columns = ["ABSOLUTE_ROWNUMBER"] + [col for col in county22.columns if col != "ABSOLUTE_ROWNUMBER"]
    county22 = county22[columns]
    return county22


def rename1(county22):
    '''
    Renames the columns in the county22 dataframe to match flight22 dataframe. 
    This promotes data integrity and data tidiness. 
    
    Args: 
        county22 (DF): Contains county weather events that surround Chicago O'Hare Airport.
        
    Returns: 
        count22 (DF): updated Dataframe with consistent column names
        
        '''
    county22 = county22.rename(columns={
        'EVENT_ID': 'EventID', 
        'CZ_NAME_STR': 'CZNameStr',
        'BEGIN_LOCATION': 'BeginLocation',
        'BEGIN_DATE': 'BeginDate',
        'BEGIN_TIME': 'BeginTime',
        'EVENT_TYPE': 'EventType',
        'MAGNITUDE': 'Magnitude',
        'TOR_F_SCALE': 'TorFScale',
        'DEATHS_DIRECT': 'DeathsDirect',
        'INJURIES_DIRECT': 'InjuriesDirect',
        'DAMAGE_PROPERTY_NUM': 'DamagePropertyNum',
        'DAMAGE_CROPS_NUM': 'DamageCropsNum',
        'STATE_ABBR': 'State', 
        'CZ_TIMEZONE': 'CZTimezome',
        'MAGNITUDE_TYPE': 'MagnitudeType',
        'EPISODE_ID': 'EpisodeID', 
        'CZTYPE': 'CZType', 
        'CZ_FIPS': 'CZFips', 
        'INJURIES_INDIRECT': 'InjuriesIndirect', 
        'DEATHS_INDIRECT': 'DeathsIndirect', 
        'SOURCE': 'Source',
        'FLOOD_CAUSE': 'FloodCause', 
        'TOR_LENGTH': 'TorLength', 
        'TOR_WIDTH': 'TorWidth', 
        'BEGIN_RANGE': 'BeginRange', 
        'BEGIN_AZIMUTH': 'BeginAzimuth', 
        'END_RANGE': 'EndRange', 
        'END_AZIMUTH': 'EndAzimuth', 
        'END_LOCATION': 'EndLocation',
        'END_DATE': 'EndDate', 
        'END_TIME': 'EndTime',
        'BEGIN_LAT': 'BeginLat', 
        'BEGIN_LOT': 'BeginLot', 
        'END_LAT': 'EndLat', 
        'END_LOT': 'EndLot', 
        'EVENT_NARRATIVE': 'EventNarrative', 
        'EPISODE_NARRATIVE': 'EpisodeNarrative', 
        'ABSOLUTE_ROWNUMBER': 'AbsoluteRowNumber'
    })
    return county22



def filter_flights(flight22): 
    '''
    Filters the flight22 dataframe for flights originating from "ORD" and where OriginCityName is "Chicago, IL"
    
    Args: 
        flight22 (DF): Contains flight information for Chicago O'Hare Airport.
        
    Returns: 
        Df: Updated dataframe. 
    '''
    fflights22 = flight22[(flight22['Origin'] == 'ORD') & (flight22['OriginCityName'] == 'Chicago, IL')]
    return fflights22

def merge(flight22, county22): 
    '''
    Merges flight22 and county22 dataframes on BeginDate and FlightDate
    
    Args: 
        county22 (DF): Contains county weather events that surround Chicago O'Hare Airport.
        flight22 (DF): Contains flight information for Chicago O'Hare Airport.
        
    Returns: 
        wf (DF): New merged dataframe with all columns from both dat asets.
    
    '''
    fflights22 = flight22.copy()
    
    fflights22.loc[:, 'FlightDate'] = pd.to_datetime(fflights22['FlightDate'])
    county22.loc[:, 'BeginDate'] = pd.to_datetime(county22['BeginDate'])
    wf = pd.merge(fflights22, county22, left_on='FlightDate', right_on='BeginDate', how='outer')

    wf.to_csv(transformed_directory / 'wf.csv', index=False)
    return wf


#def copy_wf???? I dont think i need here 

def preprocess(wf): 
    #need to fix 
    '''
    Preprocess the data for regression analysis. 
    
    Args: 
        wf (DF): The merged dataframe with weather and flight information. 
        
    Returns: 
        Df and series: the features and target variable 
    
    '''
    
    wf.fillna(0, inplace=True)
    
    #feature engineering 
    wf['season'] = pd.to
    
    #defining features and target variables 
    features = ['FlightDate','OriginCityName', 'Cancelled', 'CancellationCode', 'WeatherDelay']
    target = 'EventType'
    
    X = wf(features)
    y = wf(target)
    
    return X, y

def train_model(X_train, y_train): 
    '''
    Train the model. 
    
    Args: 
        X_train (DF): features for training 
        y_train(series): target variable for training 
    
    Returns: 
        model: trained linear regression model 
    '''
    model = LinearRegression() 
    model.fit(X_train, y_train)
    return model 

def evalaute(model, X_test, y_test): 
    '''
    Evaluates the linear regression model. 
    
    Args: 
        model: the trained linear regression model
        X_train (DF): features for training 
        y_train(series): target variable for training 
    
    Returns: 
        dict
    
    '''
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return #mean squared error and r2 

def scatter_plot_regression_results(X_test, y_test, y_pred):
    plt.figure(figsize = (10, 6))
    
    plt.scatter(y_test, y_pred, alpha=0.5, label='Data points')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('enter title here')
    


def main(): 
    county22, flight22 = load_from_extracted(extracted_directory)
    flight22 = fix_flight22(flight22)
    county22 = fix_county22(county22)
    county22 = rename1(county22)
    
    filtered_flights = filter_flights(flight22)
    
# here my def main() would be fiished
    

if __name__ == "__main__":
    main()
    



