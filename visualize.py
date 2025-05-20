import matplotlib.pyplot as plt
import train as t
import dataframe as df
import seaborn as sns
import numpy as np
import pandas as pd

num_cols, cat_cols = t.get_cols()
y_test,y_pred = t.model_pipeline(df.get_feature_boston(),df.get_target_boston(),num_cols=num_cols,cat_cols=cat_cols)

def plot_actual_vs_predicted(y_test, y_pred):
        #y_test = y_test.values.ravel()
        #y_pred = y_pred.values.ravel()
        plt.figure(figsize=(10,8))
        sns.scatterplot(x=y_test,y=y_pred)
        plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'--r',label = 'Ideal pred line')
        plt.title("Actual vs. predictions")
        plt.xlabel("Actual values")
        plt.ylabel("predicted values")
        plt.tight_layout()
        plt.show()

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals vs. Actual Values")
    plt.tight_layout()
    plt.show()