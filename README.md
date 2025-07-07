# HEART DISEASE PROJECT 
Ce projet implémente un pipeline complet de machine learning pour prédire les maladies cardiaques en utilisant des données médicales. Il comprend l'expérimentation de modèles, l'optimisation d'hyperparamètres, l'évaluation et une interface utilisateur interactive.
## 🏗️ dArchitecture du projet 
heart-disease-prediction/
├── utils.py                    # Fonctions utilitaires
├── experiment.py              # Expérimentation des modèles
├── experiment_hyper.py        # Optimisation des hyperparamètres
├── model_eval.py              # Évaluation finale du modèle
├── HD_stream.py               # Interface Streamlit
├── params.yaml                # Configuration des paramètres
├── requirements.txt           # Dépendances Python
├── heart-disease.csv          # Dataset (à ajouter)
└── README.md                  # Documentation
## Project Overview
Heart disease remains a leading cause of mortality worldwide, making early detection critical. This project applies machine learning techniques to analyze patient data and provide predictive insights, potentially aiding healthcare professionals in decision-making. The dataset includes 303 patient records with 14 attributes, and the final model aims to achieve high accuracy and interpretability.
## Dataset
The dataset is sourced from the UCI Machine Learning Repository and comprises 303 entries with the following 14 features:

- **Age**: Âge du patient
- **Sex**: Sexe (0=Femme, 1=Homme)
- **CP**: Type de douleur thoracique (0-3) (0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic).
- **Trestbps**: Pression artérielle au repos en mm Hg (e.g., 94–200).
- **Chol**: Cholestérol sérique en mg/dl (e.g., 126–564).
- **Fbs**: Glycémie à jeun > 120 mg/dl (1 = True, 0 = False).
- **Restecg**: Résultats ECG au repos (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy).
- **Thalach**: Fréquence cardiaque maximale (e.g., 71–202).
- **Exang**: Angine induite par l'exercice (1 = Yes, 0 = No).
- **Oldpeak**: Dépression ST (e.g., 0–6.2).
- **Slope**: Pente du segment ST (0 = Upsloping, 1 = Flat, 2 = Downsloping).
- **Ca**: Nombre de vaisseaux principaux (0–4) colored by fluoroscopy.
- **Thal**: Thalassémie (0 = Not specified, 1 = Normal, 2 = Fixed defect, 3 = Reversible defect).
- **Target**: Variable cible (1 = Heart disease, 0 = No heart disease).
The dataset provided includes real-world clinical measurements.
## Models Used
Several machine learning algorithms were tested for this project:

- **Logistic Regression**: A baseline linear model for binary classification.
- **Gradient boosting**: An ensemble  for interpretability and feature importance analysis.
- **HistGradientBoosting**: 
- **Random Forest**: An ensemble method to improve accuracy and reduce overfitting.
- **Support Vector Machine (SVM)**: A robust classifier for handling non-linear relationships.

Each model was evaluated using metrics such as accuracy, precision, recall, and F1-score, with hyperparameter tuning applied where applicable.
## Tracking MLflow 
Le projet utilise MLflow pour:
- Suivre les expériences et leurs paramètres
- Comparer les performances des modèles
- Enregistrer les artefacts (modèles, graphiques)
- Gérer les versions des modèles 
## Personalized prediction
The project includes an interactive feature implemented via a Streamlit application, allowing users to input custom values for each of the 13 explanatory variables (e.g., age, sex, cholesterol, etc.). A trained Logistic Regression model then estimates the probability of heart disease based on these inputs. This feature provides:
- **User Input**: A form where users can specify values for all clinical attributes.
- **Probability Output**: The model returns a percentage probability of heart disease (e.g., "73% chance of heart disease"), leveraging the linear relationship between features and the target.
- **Risk Assessment**: A simple threshold (e.g., >50%) indicates whether the risk is high or low, with visual feedback provided to the user.

This functionality enhances the practical utility of the project by enabling individualized risk assessment directly within the application, supported by the interpretability of logistic regression.

