from eda import get_preprocessing,get_categorical_cols,get_numerical_cols
from dataframe import get_boston,get_feature_boston,get_target_boston
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

def model_pipeline(feature_df,target_df,num_cols, cat_cols):
    x_train,x_test,y_train,y_test = train_test_split(feature_df,target_df,test_size=0.2,random_state=42)
    model = make_pipeline(get_preprocessing(num_cols,cat_cols),LinearRegression())
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    return y_test,y_pred

def get_cols():
    boston_df_num_cols = get_numerical_cols(get_feature_boston())
    boston_df_cat_cols = get_categorical_cols(get_feature_boston())
    boston_df_num_cols = boston_df_num_cols.drop(boston_df_cat_cols)
    return boston_df_num_cols,boston_df_cat_cols

def get_MAPE(y_test,y_pred):
    return mean_absolute_percentage_error(y_test,y_pred)