import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import importlib
import os
# LOADING DATA
def load_data():
    data = pd.read_csv('heart-disease.csv')
    X,y = data.drop('target', axis=1) , data['target']
    return X,y
def split_data(X,y):
    X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train , X_test , y_train , y_test
def load_params():
    with open('params.yaml','r') as f:
        return yaml.safe_load(f)
def get_model_class(model_config):
    module_name, class_name = model_config['class'].rsplit('.', 1)
    module = importlib.import_module(f"sklearn.{module_name}")
    return getattr(module, class_name)
    
    
    
     