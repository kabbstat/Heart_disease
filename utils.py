import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import importlib
import os
import json

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

def load_best_model():
    try:
        with open('best_model.txt') as f:
            best_model_name = f.read().strip()
        return best_model_name
    except FileNotFoundError :
        return None

def load_best_params():
    try:
        with open('best_model_params.json','r') as f:
            best_params = json.load(f)
        return best_params
    except FileNotFoundError:
        return None

def save_best_model(model_name):
    with open('best_model.txt','w') as f:
        f.write(model_name)
def save_best_params(params):
    with open('best_model_params.json','w') as f:
        json.dump(params, f, indent=2)
    
    
    
     