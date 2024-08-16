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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

data_directory = Path('data/original data')
extracted_directory = Path('data/extracted')
transformed_directory = Path('data/transformed')
loaded_directory = Path('data/loaded')
outputs_directory = Path('data/outputs')
visualizations_directory = Path('data/visualizations')

def main():
    try:
        logger.info("Starting ETL process...")

        extract_data()
        transform_data()
        load_data()

        logger.info("ETL process completed. Starting analysis...")

        df_wf = pd.read_csv(loaded_directory / 'wf.csv')
        df_wf = preprocessing1(df_wf)
        df_wf_encoded = preprocessing2(df_wf)

        df_wf.to_csv(outputs_directory / 'wf_preprocessed.csv', index=False)
        df_wf_encoded.to_csv(outputs_directory / 'wf_preprocessed_encoded.csv', index=False)

        X_train, X_test, y_train, y_test = split_regression(df_wf_encoded)
        model = build_regression(X_train, y_train)
        mse, r2, mae = test_regression(model, X_test, y_test)

        logger.info(f"Model performance - MSE: {mse}, R²: {r2}, MAE: {mae}")
        with open(outputs_directory / 'model_performance.csv', mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Mean Squared Error', 'R² Score', 'Mean Absolute Error'])
            writer.writerow([mse, r2, mae])

        y_pred = model.predict(X_test)
        plot_regression_scatter(X_test, y_test, y_pred)
        analyze_event_impact(df_wf)
        visualize_data()

        logger.info("Main script completed successfully.")

    except Exception as e:
        logger.error("An error occurred during the process", exc_info=True)

if __name__ == "__main__":
    main()
