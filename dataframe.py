import pandas as pd

def get_boston():
    boston_df = pd.read_csv("C:\\Users\\Yash\\anaconda_projects\\All Codes\\Regression\\boston.csv")
    return boston_df

def get_feature_boston():
    a = get_boston()
    a_x=a.drop(columns=['Price','Unnamed: 0'])
    return a_x

def get_target_boston():
    a = get_boston()
    a_y = a['Price']
    a_y = pd.DataFrame(a_y)
    return a_y

def get_startup_df():
    startup_df = pd.read_csv("C:\\Users\\Yash\\anaconda_projects\\All Codes\\Regression\\50_Startups.csv")
    return startup_df