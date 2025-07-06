import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import json
from utils import load_data, get_model_class, load_params
from sklearn.model_selection import GridSearchCV, StratifiedKFold
 

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("hyper_parameter")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def main():
    X,y = load_data()
    params = load_params()
    with open('best_model.txt','r') as f:
        best_model_name = f.read().strip()
    grid_config = params['experiment_hyper']['param_grid'][best_model_name]
    model_class = get_model_class(params['experiment']['models'][best_model_name])
    

    grid_search = GridSearchCV(model_class(), param_grid=grid_config, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X,y)
    with mlflow.start_run(run_name=f"Hyperparameter of {best_model_name}"):
        best_params = grid_search.best_params_
        best_score_mean = grid_search.best_score_
        best_score_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        best_model_hyper = grid_search.best_estimator_
        mlflow.log_param("best param", best_params)
        mlflow.log_metric("best score", best_score_mean)
        mlflow.log_metric("best score std", best_score_std)
        mlflow.sklearn.log_model(best_model_hyper,"best model")
        with open('best_model_params.json','w') as f:
            json.dump(best_params,f)
        mlflow.log_artifact('best_model_params.json')
if __name__ == "__main__":
    main()
        
#     pipeline = Pipeline([('scaler',StandardScaler()),
#                         ('random_forest', RandomForestClassifier() )])   
    

    
