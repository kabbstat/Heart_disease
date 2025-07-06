import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import json
from utils import load_data, get_model_class, load_params, load_best_model, load_best_params, split_data
from sklearn.model_selection import GridSearchCV, StratifiedKFold

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("model_eval")
def main():
    X,y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    model_map = { 
        'LogisticRegression': LogisticRegression}
    best_model_name = load_best_model()
    best_params = load_best_params()
    model_class = model_map.get(best_model_name)
    
    if model_class is None:
        raise ValueError(f"unknown model name {best_model_name}")
    model = model_class(**best_params)
    pipeline = Pipeline([('Scaler', StandardScaler()),
                         ('model', model)])
    scoring = ['accuracy','f1','precision','recall']
    with mlflow.start_run(run_name=f"evaluation de {best_model_name}"):
        
        cv_results = cross_validate(pipeline, X_train, y_train, cv=5, scoring=scoring ,return_train_score=False, n_jobs=-1)
        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            mlflow.log_metric(f'cv_{ metric}_mean', test_scores.mean())
            mlflow.log_metric(f'cv_{metric}_std', test_scores.std())
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test,y_pred)
        test_f1 = f1_score(y_test,y_pred)
        test_recall = recall_score(y_test,y_pred)
        test_precision = precision_score(y_test,y_pred)
        # log test metric 
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1",test_f1)
        mlflow.log_metric("test_recall",test_recall)
        mlflow.log_metric("test_precision",test_precision)
        #log trained model 
        mlflow.sklearn.log_model(pipeline, 'model')
        print("Cross-validation results:")
        for metric in scoring:
            mean = cv_results[f'test_{metric}'].mean()
            std = cv_results[f'test_{metric}'].std()
            print(f"  {metric}: {mean:.4f} +/- {std:.4f}")

        print("Test set performance:")
        print(f"  accuracy:  {test_accuracy:.4f}")
        print(f"  f1-score:  {test_f1:.4f}")
        print(f"  precision: {test_precision:.4f}")
        print(f"  recall:    {test_recall:.4f}")

if __name__ == "__main__":
    main()
        

    
    
    