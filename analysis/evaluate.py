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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 

logging.basicConfig(
    filename='analysis.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

loaded_directory = Path('../data/loaded')
outputs_directory = Path('../data/outputs')
visualization_directory = Path('../data/visualizations')

def read_csv(file_name='wf.csv'):
    '''
    Reads the 'wf.csv' file from the loaded directory.

    Returns:
        pd.DataFrame: The dataframe read from the CSV file.
    '''
    try:
        file_path = loaded_directory / file_name
        df = pd.read_csv(file_path, low_memory=False)
        logging.info("Successfully read 'wf.csv' from the loaded directory")
        return df
    except Exception as e:
        logging.error("Error occurred while reading 'wf.csv'", exc_info=True)
        raise

def preprocessing1(df): 
    columns_to_drop = [
        'Quarter', 'OriginStateName', 'OriginState', 'DestState', 'DestStateName', 
        'CarrierDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 
        'AbsoluteRowNumber', 'BeginLocation', 'TorFScale', 'DeathsDirect', 
        'InjuriesDirect', 'DamagePropertyNum', 'DamageCropsNum', 'State', 
        'CZTimezome', 'MagnitudeType', 'EpisodeID', 'CZ_TYPE', 'CZFips', 'WFO', 
        'InjuriesIndirect', 'DeathsIndirect', 'Source', 'FloodCause', 'TorLength', 
        'TorWidth', 'BeginRange', 'BeginAzimuth', 'EndRange', 'EndAzimuth', 
        'EndLocation', 'BeginLat', 'BEGIN_LON', 'EndLat', 'EventNarrative', 
        'EpisodeNarrative', 'END_LON', 'Month', 'DayofMonth', 'DayOfWeek',
        'Year', 'CZNameStr', 'Magnitude', 'EventID', 'OriginCityName', 'DestCityName',
        'FlightDate', 'DepTime', 'BeginDate', 'BeginTime', 'EndDate', 'EndTime',
    ] 
    try: 
        df.drop(columns=columns_to_drop, inplace=True)
        logging.info("The specified columns were dropped from the dataframe")
       
        # Removing rows where the cancellation code is 'A', 'C', or 'D'
        initial_row_count = df.shape[0]
        df = df[~df['CancellationCode'].isin(['A', 'C', 'D'])]
        filtered_row_count = df.shape[0]
        logging.debug(f"Filtered rows where CancellationCode is 'a', 'c', or 'd'. Rows before: {initial_row_count}, after: {filtered_row_count}")

        # Remove rows where the weather event is 'Drought' or 'Rip Current'
        initial_row_count = df.shape[0]
        df = df[~df['EventType'].isin(['Drought', 'Rip Current'])]
        filtered_row_count = df.shape[0]
        logging.debug(f"Filtered rows where EventType is 'Drought' or 'Rip Current'. Rows before: {initial_row_count}, after: {filtered_row_count}")
        
        # Moving the 'ID' column to the beginning
        id_column = df.pop('ID')
        df.insert(0, 'ID', id_column)
        
        # Resetting the 'ID' column to start at 1
        df['ID'] = range(1, len(df) + 1)
        logging.debug("Moved 'ID' column to the beginning of the dataframe and reset 'ID' to start at 1")

        logging.info("Successfully completed preprocessing1 steps")
        return df
    except Exception as e:
        logging.error("Error occurred during preprocessing1", exc_info=True)
        raise

def preprocessing2(df):
    '''
    Creates interaction terms for feature engineering and applies one-hot encoding.

    Args:
        df (pd.DataFrame): The dataframe to preprocess.

    Returns:
        pd.DataFrame: The dataframe with new interaction columns and one-hot encoded features.
    '''
    try:
        # Replace NaN values in 'CancellationCode' and 'WeatherDelay'
        df['CancellationCode'] = df['CancellationCode'].fillna(0)        
        # Creating interaction terms
        df['EventType_WeatherDelay'] = df['EventType'].astype(str) + '_' + df['WeatherDelay'].astype(str)
        

        # List of columns to one-hot encode
        categorical_features = ['EventType']

        # Verify that the columns exist
        missing_cols = [col for col in categorical_features if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns missing from DataFrame: {missing_cols}")

        # Convert columns to strings
        df[categorical_features] = df[categorical_features].astype(str)

        # One-hot encoding
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_features = encoder.fit_transform(df[categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        one_hot_encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
        
        one_hot_encoded_df['WeatherDelay']=df['WeatherDelay']
        
        logging.debug("Applied one-hot encoding to the features")
        logging.info("Successfully completed preprocessing2 steps with one-hot encoding")
        return one_hot_encoded_df
    except Exception as e:
        logging.error("Error occurred during preprocessing2", exc_info=True)
        raise
    
def split_regression(one_hot_encoded_df):
    '''
    Splits the dataframe into training and test sets.

    Args:
        df (pd.DataFrame): The dataframe to split.
        
    Returns:
        X_train (pd.DataFrame): Features for the training set.
        X_test (pd.DataFrame): Features for the test set.
        y_train (pd.Series): Target for the training set.
        y_test (pd.Series): Target for the test set
    '''
    try: 
        X = one_hot_encoded_df.drop(['WeatherDelay'], axis=1) 
        y = one_hot_encoded_df['WeatherDelay']
        
                # Ensure no NaN values in the target variable
        if y.isnull().any():
            y = y.fillna(0)
            logging.debug("Replaced NaN values in target variable 'WeatherDelay' with 0")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        logging.info("The data was successfully split for the regression model")
        return X_train, X_test, y_train, y_test
    except Exception as e: 
        logging.error("Error when splitting data for regression", exc_info=True)
        raise
    
def build_regression(X_train, y_train): 
    '''
    Builds the linear regression model.
    
    Args: 
        X_train (DF): Features 
        y_train (DF): Target 
        
    Returns: 
        Linear Regression Model 
    '''
    try: 
        model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info("Linear regression model was trained successfully")
        return model 
    except Exception as e: 
        logging.error("Error when building the regression model", exc_info=True)
        raise 
    
def test_regression(model, X_test, y_test):
    '''
    Tests the linear regression model.
    
    Args: 
        model (LinearRegression):
        X_test (DF): Features
        y_test (DF): Target
        
    Returns: 
        mse (float): Mean Squared Error of the model on the test set
        r2 (float): R2 score of the model on the test set
        
    '''
    try: 
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        logging.info(F" The model returned - MSE: {mse}, R2: {r2}, MAE: {mae}")
        return mse, r2, mae
    except Exception as e: 
        logging.error("Error occured when testing the regression model", exc_info=True)
        raise 
def plot_regression_scatter(X_test, y_test, y_pred):
    '''
    Plots a scatter plot with regression line.
    
    Args:
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Actual target values
        y_pred (np.array): Predicted target values
    '''
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='w', s=60)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Diagonal line for perfect prediction
        plt.xlabel('Actual Weather Delay')
        plt.ylabel('Predicted Weather Delay')
        plt.title('Scatter Plot of Actual vs. Predicted Weather Delay')
        plt.grid(True)
        plt.savefig(visualization_directory / 'regression_scatterplot.png')  # Save the plot as a PNG file
        plt.show()
        logging.info("Regression scatter plot saved as 'regression_scatterplot.png'")
    except Exception as e:
        logging.error("Error occurred while plotting the regression scatter plot", exc_info=True)
        raise
        
                
'''
visualization here 
'''
'''
talk through a case
'''
    
def main(): 
    try: 
        
        df_wf = read_csv()
        print("Data read successfully")
        print(df_wf.head())
        
        # Apply preprocessing1
        df_wf = preprocessing1(df_wf)
        print("Completed preprocessing1")
        print(df_wf.head())
        
        # Save the intermediate dataframe after preprocessing1
        output_file_path_intermediate = outputs_directory / 'wf_preprocessed.csv'
        df_wf.to_csv(output_file_path_intermediate, index=False)
        logging.info(f"Successfully saved the dataframe after preprocessing1 to {output_file_path_intermediate}")
        print(f"Data after preprocessing1 saved to {output_file_path_intermediate}")
        
        # Apply preprocessing2 with one-hot encoding
        df_wf_encoded = preprocessing2(df_wf)
        print("Completed preprocessing2")
        print(df_wf_encoded.head())  # Check the dataframe after preprocessing2
        
        output_file_path_encoded = outputs_directory / 'wf_preprocessed_encoded.csv'
        df_wf_encoded.to_csv(output_file_path_encoded, index=False)
        logging.info(f"Successfully saved the preprocessed and encoded dataframe to {output_file_path_encoded}")
        print(f"Data saved to {output_file_path_encoded}")
        
        # Split the data for regression
        X_train, X_test, y_train, y_test = split_regression(df_wf_encoded)
        
        # Build the regression model
        model = build_regression(X_train, y_train)
        
        # Test the regression model
        mse, r2, mae = test_regression(model, X_test, y_test)
        
        print(f"Mean Squared Error: {mse}")
        print(f"R² Score: {r2}")
        print(f"Mean Absolute Error: {mae}")
        
        # Save regression metrics to a new CSV file
       # Save the performance metrics to a CSV file
        metrics_file_path = outputs_directory / 'model_performance.csv'
        with open(metrics_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Mean Squared Error', 'R² Score', 'Mean Absoulte Error'])
            writer.writerow([mse, r2, mae])
        
        logging.info(f"Model performance metrics saved to {metrics_file_path}")
        print(f"Mean Squared Error: {mse}")
        print(f"R² Score: {r2}")
        print(f"Mean Absolute Error: {mae}")
        
        # Plot and save the regression scatter plot
        y_pred = model.predict(X_test)
        plot_regression_scatter(X_test, y_test, y_pred)
        
    except Exception as e:
        logging.error("Error occurred in the main function", exc_info=True)
        raise
    
if __name__ == "__main__":
    main()