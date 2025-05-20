import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def get_numerical_cols(data_features):
    col = data_features.select_dtypes(include='number').columns
    return col

def get_categorical_cols(data_features):
    col = [col for col in data_features.columns if data_features[col].dtype=='object' or data_features[col].nunique()<10]
    return col

def get_missing_values(data):
    return data.isnull().sum()

def get_num_transformer():
    return Pipeline(steps=[('Imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])

def get_cat_transformer():
    return Pipeline(steps=[('Imputer',SimpleImputer(strategy='most_frequent')),('Encoder',OneHotEncoder(handle_unknown='ignore'))])

def get_preprocessing(num_cols, cat_cols):
    return ColumnTransformer(transformers=[('num',get_num_transformer(),num_cols),('cat',get_cat_transformer(),cat_cols)])

def get_scatter_plot(data,x,y):
    plt.figure(figsize=(20,8))
    sns.scatterplot(data,x=x,y=y)
    plt.show()

def get_kde_plot(data,x,y):
    plt.figure(figsize=(20,8))
    sns.kdeplot(data=data, x=x,y=y,fill=True)
    plt.show()

def get_heatmap(data):
    plt.figure(figsize=(20,8))
    correlation = data.corr()
    sns.heatmap(correlation,annot = True,fmt=".2f",cmap='coolwarm')
    plt.title("Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()  
    
def get_lmplot(data,feature_numeric_data,target_column):
    for i in feature_numeric_data:
        sns.lmplot(data=data,x=i,y=target_column)
        plt.show()

def get_pairgrid(data,feature_data,target_data):
    plt.figure(figsize=(20,8))
    g=sns.PairGrid(data=data,x_vars=feature_data,y_vars=target_data)
    g.map(sns.regplot)
    plt.show()

def get_countplot(data,feature_data,hue):
    plt.figure(figsize=(20,8))
    for i,el in enumerate(feature_data):
        sns.countplot(data=data,x=el,hue=hue)
        plt.show()


    