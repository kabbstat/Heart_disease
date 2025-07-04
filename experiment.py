import pandas as pd 
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
import mlflow
from utils import load_data, load_params, get_model_class
import mlflow.sklearn 

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("1er_experimentation")

def main():
    params = load_params()
    models_config = params['experiment']['models']
    X, y  = load_data() 
    best_score = 0
    best_model = None
    
    with mlflow.start_run(run_name="experimentation pour classification"):
        for model_name, model_config in models_config.items():
            with mlflow.start_run(run_name=f"experimentation de {model_name}", nested=True):
                model_class = get_model_class(model_config)
                model = model_class(**model_config.get('params',{}))
                cv = cross_validate(model, X, y, cv=params['experiment']['cv'])
                mean_accuracy = cv['test_score'].mean()
                std_accuracy = cv['test_score'].std()
                mlflow.log_param("model name", model_name)
                mlflow.log_params(model_config.get('params',{}))
                mlflow.log_metric("mean_accuracy", mean_accuracy)
                mlflow.log_metric("std_accuracy", std_accuracy)
                model.fit(X,y)
                mlflow.sklearn.log_model(model, model_name)
                if mean_accuracy > best_score:
                    best_score, best_model  = mean_accuracy, model_name
        if best_model :
            mlflow.set_tag("best model", best_model)
            mlflow.log_param("best model", best_model)
            mlflow.log_metric("best score", best_score)
        with open('best_model.txt','w') as f:
            f.write(best_model)
if __name__ == "__main__":
    main()
                    
                
                

