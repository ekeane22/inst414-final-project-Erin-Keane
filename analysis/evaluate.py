'''
The type of analytics I am performing is descriptive analytics. 

The pipeline for descriptive analytics is: 
1. Business Problem, Project Design, Data 
2. ETL 
3. Exploration and Visualization --> Aggregation and Descriptive Statistics 
4. Visualization and Presentation

What analytical models am I doing based on my business problem and datasets?

Measures of Central Tendency: Mean, Median, Mode
Measures of Frequency: Count, Frequency
Measures of Dispersion/Variation: Variance and Standard Deviation
'''

# I would load in the dataframe from the processed.py that holds the cleaned and combined
# dataframes. below is all rough code for later ideas. 

    
def central_ten(#dataframe, #column):
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode()
    return  
    

def frequency(df, column):
    '''
    count, frequency
    '''            

    
def dispersion(df, column):
    '''
    variance and std 
    similiar format as above but with .var and .std
    '''


