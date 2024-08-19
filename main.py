import pandas as pd
import numpy as np
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from etl.extract import process_data as extract_data
from etl.transform import main as transform_data 
from etl.load import main as load_data 
from analysis.evaluate import (
    preprocessing1, preprocessing2, split_regression, build_regression, 
    test_regression, plot_regression_scatter, analyze_event_impact, regression_analysis_impact
)
from vis.visualizations import main as visualize_data


logging.basicConfig(
    filename='main.log',
    filemode='w',
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#Define data directories 
data_directory = Path('data_ek/data/original_data')
extracted_directory = Path('data_ek/data/extracted')
transformed_directory = Path('data_ek/data/transformed')
loaded_directory = Path('data_ek/data/loaded')
outputs_directory = Path('data_ek/data/outputs')
visualizations_directory = Path('data_ek/data/visualizations')

def main():
    '''
    Main function to execute the ETL process, perform data analysis, and generate visualizations.

    This function orchestrates the extraction, transformation, and loading of data,
    performs data preprocessing and regression analysis, and generates relevant visualizations.
    '''
    try:
        logger.info("Starting ETL process...")
        # Execute ETL steps
        extract_data()
        transform_data()
        load_data()

        logger.info("ETL process completed. Starting analysis...")

        # Load preprocessed data
        df_wf = pd.read_csv(loaded_directory / 'wf.csv', low_memory=False)
        
        # Preprocessed data
        df_wf = preprocessing1(df_wf)
        df_wf_encoded = preprocessing2(df_wf)
        
        # Save processed data
        df_wf.to_csv(outputs_directory / 'wf_preprocessed.csv', index=False)
        df_wf_encoded.to_csv(outputs_directory / 'wf_preprocessed_encoded.csv', index=False)
        
        # Split data for regression analysis 
        X_train, X_test, y_train, y_test = split_regression(df_wf_encoded)
        # Build and test the regression model
        model = build_regression(X_train, y_train)
        mse, r2, mae = test_regression(model, X_test, y_test)
        # log model performance metrics 
        logger.info(f"Model performance - MSE: {mse}, R²: {r2}, MAE: {mae}")
        
        # Save the model performance metrics to a CSV
        with open(outputs_directory / 'model_performance.csv', mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Mean Squared Error', 'R² Score', 'Mean Absolute Error'])
            writer.writerow([mse, r2, mae])
        
        # Generate and save regression scatter plot
        y_pred = model.predict(X_test)
        plot_regression_scatter(X_test, y_test, y_pred)
        
        #Genereate and save event impact analysis
        analyze_event_impact(df_wf)
        regression_analysis_impact(df_wf_encoded)
    
        # Generate EDA visualizations
        visualize_data()

        logger.info("Main script completed successfully.")

    except Exception as e:
        logger.error("An error occurred during the process", exc_info=True)

if __name__ == "__main__":
    main()
