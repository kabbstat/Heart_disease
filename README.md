Projet de Prédiction des Maladies Cardiaques
Date: 09 juillet 2025

🎯 Introduction
Ce projet implémente un pipeline complet de machine learning pour prédire les maladies cardiaques à partir de données médicales. Il inclut l'expérimentation de modèles, l'optimisation des hyperparamètres, l'évaluation des performances et une interface utilisateur interactive avec Streamlit. L'objectif est d'offrir un outil d'aide à la décision pour les professionnels de santé, avec une précision élevée et une interprétabilité des résultats.

📂 Architecture du Projet
heart-disease-prediction/
├── data/                       # Données (heart-disease.csv)
├── src/                        # Code source
│   ├── utils.py                # Fonctions utilitaires
│   ├── experiment.py           # Expérimentation des modèles
│   ├── experiment_hyper.py     # Optimisation des hyperparamètres
│   ├── model_eval.py           # Évaluation finale du modèle
│   ├── HD_stream.py            # Interface Streamlit
├── params.yaml                 # Configuration des paramètres
├── dvc.yaml                    # Configuration DVC pour le pipeline
├── requirements.txt            # Dépendances Python
└── README.md                   # Documentation


📊 Dataset
Le dataset, issu du UCI Machine Learning Repository, contient 303 entrées avec 14 attributs médicaux :

age : Âge du patient
sex : Sexe (0 = Femme, 1 = Homme)
cp : Type de douleur thoracique (0-3)
trestbps : Pression artérielle au repos (mm Hg)
chol : Cholestérol sérique (mg/dl)
fbs : Glycémie à jeun > 120 mg/dl (1 = Vrai, 0 = Faux)
restecg : Résultats ECG au repos (0-2)
thalach : Fréquence cardiaque maximale
exang : Angine induite par l’exercice (0 = Non, 1 = Oui)
oldpeak : Dépression ST
slope : Pente du segment ST (0-2)
ca : Nombre de vaisseaux principaux (0-4)
thal : Thalassémie (0-3)
target : Variable cible (0 = Pas de maladie, 1 = Maladie)

Prétraitement : Nettoyage des données, gestion des valeurs manquantes, et encodage des variables catégoriques.

🧠 Pipeline ML
1. utils.py - Fonctions Utilitaires

Chargement et prétraitement des données
Gestion des paramètres via params.yaml
Sauvegarde/chargement des modèles et résultats

2. experiment.py - Expérimentation

Test de 5 modèles : Random Forest, Gradient Boosting, HistGradientBoosting, SVM, Logistic Regression
Validation croisée et suivi via MLflow
Sélection du meilleur modèle (enregistré dans best_model.txt)

3. experiment_hyper.py - Optimisation

Grid Search pour optimiser les hyperparamètres du meilleur modèle
Validation croisée stratifiée
Sauvegarde des paramètres optimisés en JSON

4. model_eval.py - Évaluation

Évaluation du modèle optimisé avec métriques (accuracy, precision, recall, F1, AUC)
Visualisations : matrice de confusion, courbe ROC
Validation croisée sur l’ensemble complet

5. HD_stream.py - Interface Streamlit

Exploration des données (EDA) avec statistiques et visualisations
Test interactif des modèles
Prédiction personnalisée avec probabilité et évaluation du risque


📈 Suivi avec MLflow
MLflow est utilisé pour :

Suivi des expériences, paramètres et métriques
Comparaison des performances des modèles
Enregistrement des artefacts (modèles, graphiques)
Gestion des versions

Accès : http://127.0.0.1:5000 après lancement du serveur MLflow.

🔮 Prédiction Personnalisée
L’interface Streamlit permet :

Saisie : Formulaire pour entrer les 13 attributs cliniques
Prédiction : Probabilité de maladie cardiaque (ex. : "73% de risque")
Visualisation : Indicateur de risque (>50% = élevé) avec retour visuel
EDA : Statistiques, histogrammes, matrice de corrélation


🚀 Installation et Utilisation
Prérequis

Python 3.8+
DVC pour la gestion du pipeline
MLflow pour le suivi des expériences

Étapes

Cloner le dépôt :
git clone https://github.com/votre-utilisateur/heart-disease-prediction.git
cd heart-disease-prediction


Installer les dépendances :
pip install -r requirements.txt


Initialiser DVC :
dvc init
dvc pull


Lancer le serveur MLflow :
mlflow server --host 127.0.0.1 --port 5000


Exécuter le pipeline 
dvc repro


Lancer l’interface Streamlit :
streamlit run src/HD_stream.py




📝 Exemple d’Utilisation

Exécutez dvc repro pour entraîner et évaluer les modèles.
Lancez Streamlit avec streamlit run src/HD_stream.py.
Entrez des valeurs cliniques (ex. : âge = 55, sexe = 1, chol = 240).
Obtenez une prédiction (ex. : "73% de risque") et visualisez le niveau de risque.


⚠️ Avertissements

Usage éducatif : Ce projet ne remplace pas un diagnostic médical.
Interprétation : Les prédictions doivent être validées par un professionnel de santé.


🔧 Améliorations Apportées

Pipeline DVC : Automatisation du workflow ML.
Code modulaire : Séparation des responsabilités pour une meilleure maintenabilité.
Interface utilisateur : Visualisations interactives et recommandations contextuelles.
Documentation : Instructions claires et complètes.


🔮 Perspectives

Intégrer des modèles plus avancés (ex. : réseaux neuronaux).
Ajouter des visualisations avancées (ex. : SHAP pour l’interprétabilité).
Déployer l’application sur un serveur cloud.


👤 Auteur
KABBAJ MOHAMED  

Développé avec Streamlit, Scikit-learn, MLflow et DVC.  
Contact : votre-email@example.com


📄 Licence
Sous licence MIT. Voir LICENSE.
