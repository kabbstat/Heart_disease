import pandas as pd
import mlflow
import mlflow.sklearn
from utils import load_data, get_model_class, load_params, load_best_model, save_best_params
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import os
 
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("hyper_parameter")



def main():
    X,y = load_data()
    params = load_params()
    best_model_name = load_best_model()
    if not best_model_name:
        print("pas de meilleure modele afficher")
    grid_config = params['experiment_hyper']['param_grid'][best_model_name]
    model_class = get_model_class(params['experiment']['models'][best_model_name])
    cv = StratifiedKFold(n_splits=params['experiment_hyper']['cv'], shuffle=True, random_state=42)

    grid_search = GridSearchCV(model_class(), param_grid=grid_config, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X,y)
    with mlflow.start_run(run_name=f"Hyperparameter of {best_model_name}"):
        best_params = grid_search.best_params_
        best_score_mean = grid_search.best_score_
        best_score_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        best_model_hyper = grid_search.best_estimator_
        mlflow.log_param("best model", best_model_name)
        mlflow.log_params(best_params)
        mlflow.log_metric("best score", best_score_mean)
        mlflow.log_metric("best score std", best_score_std)
        mlflow.sklearn.log_model(best_model_hyper,"best model tuned")
        save_best_params(best_params)
        mlflow.log_artifact("best_model_params.json")
        print(f"meilleure parametre du modele {best_params}")
        print(f"meilleure score cv {best_score_mean:.4f}+\-{best_score_std:.4f}")
if __name__ == "__main__":
    main()