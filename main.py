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
    test_regression, plot_regression_scatter, analyze_event_impact
)
from vis.visualizations import main as visualize_data

logging.basicConfig(
    filename='main.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

data_directory = Path('../data/original data')
extracted_directory = Path('../data/extracted')
transformed_directory = Path('../data/transformed')
loaded_directory = Path('../data/loaded')
outputs_directory = Path('../data/outputs')
visualizations_directory = Path('../data/visualizations')


def main():
    try: 
        logging.info("Starting data extraction...")
        extract_data()
        
        logging.info("Starting data transformation...")
        transform_data()
        
        logging.info("Starting data loading...")
        load_data()
        
        # Read and preprocess data
        df_wf = pd.read_csv(loaded_directory / 'wf.csv')
        df_wf = preprocessing1(df_wf)
        df_wf_encoded = preprocessing2(df_wf)
        
        # Save intermediate and processed data
        output_file_path_intermediate = outputs_directory / 'wf_preprocessed.csv'
        df_wf.to_csv(output_file_path_intermediate, index=False)
        
        output_file_path_encoded = outputs_directory / 'wf_preprocessed_encoded.csv'
        df_wf_encoded.to_csv(output_file_path_encoded, index=False)
        
        # Split data and build model
        X_train, X_test, y_train, y_test = split_regression(df_wf_encoded)
        model = build_regression(X_train, y_train)
        mse, r2, mae = test_regression(model, X_test, y_test)
        
        # Print and log metrics
        print(f"Mean Squared Error: {mse}")
        print(f"R² Score: {r2}")
        print(f"Mean Absolute Error: {mae}")
        
        metrics_file_path = outputs_directory / 'model_performance.csv'
        with open(metrics_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Mean Squared Error', 'R² Score', 'Mean Absolute Error'])
            writer.writerow([mse, r2, mae])
        
        logging.info(f"Model performance metrics saved to {metrics_file_path}")
        
        # Plot regression scatter
        y_pred = model.predict(X_test)
        plot_regression_scatter(X_test, y_test, y_pred)
        
        # Analyze event impact
        analyze_event_impact(df_wf)
        
        visualize_data()

    except Exception as e:
        logging.error("An error occurred during the process", exc_info=True)

if __name__ == "__main__":
    main()
        