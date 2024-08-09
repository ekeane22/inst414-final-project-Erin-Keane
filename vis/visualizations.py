'''
Bar Charts, Line Graph, Histograms, Scatte plots.

Can be exploratory and rough. 

I saw a picture of a scatter_geo based on clusers, might try. 

I think im going to revise and build different dataframes 
based on the graph I want to make... Maybe.  
'''

import plotly 
import bokeh 
import matplotlib.pyplot as plt 
import seaborn as sns 


def scatter_plot_regression_results(X_test, y_test, y_pred):
    plt.figure(figsize = (10, 6))
    
    plt.scatter(y_test, y_pred, alpha=0.5, label='Data points')
    
    plt.plot([])
    

