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

logger = logging.getLogger(__name__)

loaded_directory = Path('data_ek/data/loaded')
outputs_directory = Path('data_ek/dataoutputs')
visualization_directory = Path('data_ek/data/visualizations')

def read_csv(file_name='wf.csv'):
    '''
    Reads the 'wf.csv' file from the loaded directory.
    
    Args: 
        file_name (str): Name of the CSV file to read. Defaults to 'wf.csv'.
    Returns:
        DF: The dataframe read from the CSV file.
        
    Raises:
        Exception: For errors encountered while reading the CSV file.
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
    '''
    Preprocesses the dataframe by dropping unnecessary columns, filtering rows, and adjusting the 'ID' column.

    Args:
        DF: The dataframe to preprocess.

    Returns:
        DF: The preprocessed dataframe.

    Raises:
        Exception: For errors encountered during preprocessing.
    
    '''
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
        DF: The dataframe to preprocess.

    Returns:
        DF: The dataframe with new interaction columns and one-hot encoded features.
    Raises: 
        KeyError: If expected columns for one-hot encoding are missing.
        Exception: For other errors encountered during preprocessing.
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
        
        one_hot_encoded_df['WeatherDelay'] = df['WeatherDelay']
        
        logging.debug("Applied one-hot encoding to the features")
        logging.info("Successfully completed preprocessing2 steps with one-hot encoding")
        return one_hot_encoded_df
    except Exception as e:
        logging.error("Error occurred during preprocessing2", exc_info=True)
        raise
    
def split_regression(one_hot_encoded_df):
    '''
    Splits the dataframe into training and test sets for regression.

    Args:
        one_hot_encoded_df (DF): The dataframe with one-hot encoded features to split.
        
    Returns:
        X_train (pd.DataFrame): Features for the training set.
        X_test (pd.DataFrame): Features for the test set.
        y_train (pd.Series): Target for the training set.
        y_test (pd.Series): Target for the test set.
        
    Raises:
        Exception: For errors encountered during the splitting process.
    '''
    try: 
        
        # Drop rows with 0 or NaN values in 'WeatherDelay'
        one_hot_encoded_df = one_hot_encoded_df[one_hot_encoded_df['WeatherDelay'].notnull()]
        one_hot_encoded_df = one_hot_encoded_df[one_hot_encoded_df['WeatherDelay'] != 0]
        logging.debug("Dropped rows with 0 or NaN values in 'WeatherDelay'")
        X = one_hot_encoded_df.drop(['WeatherDelay'], axis=1) 
        y = one_hot_encoded_df['WeatherDelay']
        
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
        X_train (DF): Training features. 
        y_train (DF): Training target.
        
    Returns: 
        Linear Regression Model 
        
    Raises:
        Exception: For errors encountered during model building and training.
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
        model (LinearRegression): The trained linear regression model.
        X_test (DF): Test features.
        y_test (DF): Test target.
        
    Returns: 
        mse (float): Mean Squared Error of the model on the test set.
        r2 (float): R2 score of the model on the test set.
        mae (float): Mean Absolute Error of the model on the test set.
    
    Raises:
        Exception: For errors encountered during model testing.
    '''
    try: 
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        logging.info(f"The model returned - MSE: {mse}, R2: {r2}, MAE: {mae}")
        return mse, r2, mae
    except Exception as e: 
        logging.error("Error occurred when testing the regression model", exc_info=True)
        raise 
def plot_regression_scatter(X_test, y_test, y_pred):
    '''
    Plots a scatter plot with regression line.
    
    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Actual target values.
        y_pred (np.array): Predicted target values.
    '''
    try:
        plt.figure(figsize=(14, 8))
        
        # Scatter plot of actual vs. predicted values
        sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='w', s=60)

        # Plot the regression line based on predictions
        sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', line_kws={"lw":2, "linestyle":"--"})
        
        plt.xlabel('Actual Weather Delay')
        plt.ylabel('Predicted Weather Delay')
        plt.title('Scatter Plot of Actual vs. Predicted Weather Delay')
        plt.grid(True)
        plt.savefig(visualization_directory / 'regression_scatterplot.png')
        plt.show()
        
        logging.info("Regression scatter plot saved as 'regression_scatterplot.png'")
    except Exception as e:
        logging.error("Error occurred while plotting the regression scatter plot", exc_info=True)
        raise
    
def regression_analysis_impact(df):
    '''
    Performs regression analysis to quantify the impact of different weather events on weather delays.
    
    Args:
        df (DF): The dataframe with one-hot encoded event types and weather delays.    
    '''
    
    try:
        
        # Clean column names to remove 'EventType_' prefix and handle 'EventType_nan'
        df = df.rename(columns=lambda x: x.replace('EventType_', '') if 'EventType_' in x else x)
        
        # Drop any columns that might have been created for 'NaN' values in one-hot encoding
        df = df.loc[:, ~df.columns.str.contains('nan', case=False)]
        # Extract features and target variable
        
        X = df.drop(columns=['WeatherDelay'])
        y = df['WeatherDelay']
        
        # Drop rows where 'WeatherDelay' or any feature is NaN
        df_clean = df.dropna()
        X_clean = df_clean.drop(columns=['WeatherDelay'])
        y_clean = df_clean['WeatherDelay']
        
        # Build the regression model
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        
        # Get coefficients and feature names
        coefficients = model.coef_
        feature_names = X.columns
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        
        # Sort by absolute value of coefficient
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
        
        # Plot the coefficients
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Feature', y='Coefficient', data=coef_df, palette='viridis', hue=None)
        plt.xticks(rotation=90)
        plt.xlabel('Weather Event Type')
        plt.ylabel('Coefficient')
        plt.title('Impact of Different Weather Events on Weather Delay (Regression Coefficients)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(visualization_directory / 'regression_impact_analysis.png')  
        plt.show()
        logging.info("Regression impact analysis plot saved as 'regression_impact_analysis.png'")
    except Exception as e:
        logging.error("Error occurred while performing regression impact analysis", exc_info=True)
        raise


def analyze_event_impact(df):
    '''
    Analyzes the impact of different weather events on weather delays and plots the results.

    Args:
        df (DF): The dataframe with event types and weather delays.
    '''
    try:
        df_clean = df.dropna(subset=['EventType', 'WeatherDelay'])
        
        # Ensure there are no 'nan' values in 'EventType'
        df_clean = df_clean[df_clean['EventType'] != 'nan']
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='EventType', y='WeatherDelay', data=df_clean)
        plt.xticks(rotation=90)
        plt.xlabel('Weather Event Type')
        plt.ylabel('Weather Delay (minutes)')
        plt.title('Impact of Different Weather Events on Weather Delay')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(visualization_directory / 'event_impact_analysis.png')  
        plt.show()
        logging.info("Event impact analysis plot saved as 'event_impact_analysis.png'")
    except Exception as e:
        logging.error("Error occurred while analyzing event impact", exc_info=True)
        raise


def main(): 
    '''
    Main function to run the data analysis pipeline, including reading data, preprocessing, 
    encoding, training, and evaluating the regression model, and plotting results.
    '''
    try: 
        
        # Create outputs and visualization directories if they don't exist
        outputs_directory.mkdir(parents=True, exist_ok=True)
        visualization_directory.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Created directories: {outputs_directory} and {visualization_directory}") 
        
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
        print(df_wf_encoded.head())  
        
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
        metrics_file_path = outputs_directory / 'model_performance.csv'
        with open(metrics_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Mean Squared Error', 'R² Score', 'Mean Absolute Error'])
            writer.writerow([mse, r2, mae])
        
        logging.info(f"Model performance metrics saved to {metrics_file_path}")
        print(f"Mean Squared Error: {mse}")
        print(f"R² Score: {r2}")
        print(f"Mean Absolute Error: {mae}")
        
        # Plot and save the regression scatter plot
        y_pred = model.predict(X_test)
        plot_regression_scatter(X_test, y_test, y_pred)
        # Analyze event impact
        analyze_event_impact(df_wf)
        
        # Perform regression analysis impact
        regression_analysis_impact(df_wf_encoded)
       
        
    except Exception as e:
        logging.error("Error occurred in the main function", exc_info=True)
        raise
if __name__ == "__main__":
    main()