import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from utils import load_data, get_model_class, load_params, load_best_model, load_best_params, split_data

#tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
#mlflow.set_tracking_uri(tracking_uri)
mlflow.set_tracking_uri("file:///tmp/mlruns")
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("model_eval")
def plot_confusion_matrix(y_true, y_pred, model_name):
    """Crée et sauvegarde la matrice de confusion"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'], 
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return f'confusion_matrix_{model_name}.png'
def plot_roc_curve(y_true, y_proba, model_name):
    """Crée et sauvegarde la courbe ROC"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title(f'Courbe ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return f'roc_curve_{model_name}.png', auc_score
def main():
    X,y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    params = load_params()
    # charger le meilleure model et ses parametres
    best_model_name = load_best_model()
    best_params = load_best_params()
    if best_model_name is None:
        raise ValueError(f"unknown model name {best_model_name}")
    print(f"evaluation du modele {best_model_name}")
    print(f"avec les parametres {best_params}")
    model_class = get_model_class(params["experiment"]["models"][best_model_name])
    if best_params:
        model = model_class(**best_params)
    else:
        model = model_class(**params["experiment"]["models"][best_model_name].get('params', {}))
    #pipeline = Pipeline([('Scaler', StandardScaler()),
     #                    ('model', model)])
    model.fit(X_train,y_train) 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict proba") else None
    # calcul des metriques
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred, output_dict=True) 
    scoring = ['accuracy','f1','precision','recall']
    cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring ,return_train_score=False, n_jobs=-1)

    with mlflow.start_run(run_name=f"evaluation de {best_model_name}"):
        # log parameter
        mlflow.log_param("model name", best_model_name)
        if best_params:
            mlflow.log_params(best_params)
        # log des metrics de test
        mlflow.log_metric("test accuracy", accuracy)
        mlflow.log_metric("test precision", report['1']['precision'])
        mlflow.log_metric("test f1 score", report['1']['f1-score'])
        mlflow.log_metric("test recall", report['1']['recall'])
        
        mlflow.log_metric('cv_accuracy_mean', cv_results['test_accuracy'].mean())
        mlflow.log_metric('cv_accuracy_std', cv_results['test_accuracy'].std())
        mlflow.log_metric('cv_precision_mean', cv_results['test_precision'].mean())
        mlflow.log_metric('cv_recall_mean', cv_results['test_recall'].mean())
        mlflow.log_metric('cv_f1_mean', cv_results['test_f1'].mean())
        # ROC AUC si possible
        if y_prob is not None:
            auc_score = roc_auc_score(y_test,y_prob)
            mlflow.log_metric("test auc", auc_score)
            roc_file,_ = plot_roc_curve(y_test, y_prob,best_model_name)
            mlflow.log_artifact(roc_file)
        cm_file = plot_confusion_matrix(y_test,y_pred, best_model_name)
        mlflow.log_artifact(cm_file)
        
        mlflow.sklearn.log_model(model,"final model")
        # log test metric 
 # Afficher les résultats
        print(f"\n=== RÉSULTATS DE L'ÉVALUATION ===")
        print(f"Accuracy sur test: {accuracy:.4f}")
        print(f"Précision: {report['1']['precision']:.4f}")
        print(f"Rappel: {report['1']['recall']:.4f}")
        print(f"F1-score: {report['1']['f1-score']:.4f}")
        if y_prob is not None:
            print(f"AUC: {auc_score:.4f}")
        
        print(f"\n=== VALIDATION CROISÉE ===")
        print(f"Accuracy CV: {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
        print(f"Précision CV: {cv_results['test_precision'].mean():.4f} (+/- {cv_results['test_precision'].std():.4f})")
        print(f"Rappel CV: {cv_results['test_recall'].mean():.4f} (+/- {cv_results['test_recall'].std():.4f})")
        print(f"F1-score CV: {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")

if __name__ == "__main__":
    main()
        

    
    
    