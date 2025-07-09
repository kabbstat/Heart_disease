# HEART DISEASE PROJECT 
Ce projet implémente un pipeline complet de machine learning pour prédire les maladies cardiaques en utilisant des données médicales. Il comprend l'expérimentation de modèles, l'optimisation d'hyperparamètres, l'évaluation et une interface utilisateur interactive.
## 📂 dArchitecture du projet 
heart-disease-prediction/
├── utils.py                    # Fonctions utilitaires
├── experiment.py               # Expérimentation des modèles
├── experiment_hyper.py         # Optimisation des hyperparamètres
├── model_eval.py               # Évaluation finale du modèle
├── HD_stream.py                # Interface Streamlit
├── params.yaml                 # Configuration des paramètres
├── requirements.txt            # Dépendances Python
├── heart-disease.csv           # Dataset (à ajouter)
└── README.md                   # Documentation                
## Project Overview
Heart disease remains a leading cause of mortality worldwide, making early detection critical. This project applies machine learning techniques to analyze patient data and provide predictive insights, potentially aiding healthcare professionals in decision-making. The dataset includes 303 patient records with 14 attributes, and the final model aims to achieve high accuracy and interpretability.
## 📊 Dataset
The dataset is sourced from the UCI Machine Learning Repository and comprises 303 entries with the following 14 features:

- **Age**: Âge du patient
- **Sex**: Sexe (0=Femme, 1=Homme)
- **CP**: Type de douleur thoracique (0-3) (0 = Angine typique, 1 = Angine atypique, 2 = Douleur non angineuse, 3 = Asymptomatique).
- **Trestbps**: Pression artérielle au repos (mm Hg)
- **Chol**: Cholestérol sérique (mg/dl)
- **Fbs**: Glycémie à jeun > 120 mg/dl (1 = Vrai, 0 = Faux)
- **Restecg**:  Résultats ECG au repos (0 = Normal, 1 = Anomalie onde ST-T, 2 = Hypertrophie ventriculaire gauche)
- **Thalach**: Fréquence cardiaque maximale
- **Exang**: Angine induite par l'exercice (1 = Oui, 0 = Non).
- **Oldpeak**: Dépression ST induite par l’exercice
- **Slope**: Pente du segment ST (0 = En montée, 1 = Plat, 2 = En descente)
- **Ca**: Nombre de vaisseaux principaux (0-4) colorés par fluoroscopie
- **Thal**: Thalassémie (0 = Non spécifié, 1 = Normal, 2 = Défaut fixe, 3 = Défaut réversible)
- **Target**: Variable cible (1 = Maladie cardiaque, 0 = Pas de maladie cardiaque).
The dataset provided includes real-world clinical measurements.
# Projet de Prédiction des Maladies Cardiaques

Ce projet implémente un pipeline complet de machine learning pour prédire les maladies cardiaques en utilisant des données médicales. Il comprend l'expérimentation de modèles, l'optimisation d'hyperparamètres, l'évaluation et une interface utilisateur interactive.

## 🏗️ Architecture du Projet

```
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
```

## 📊 Pipeline ML

### 1. **utils.py** - Fonctions Utilitaires
- Chargement et division des données
- Gestion des paramètres YAML
- Sauvegarde/chargement des résultats
- Instanciation dynamique des modèles

### 2. **experiment.py** - Expérimentation des Modèles
- Test de 5 modèles différents avec validation croisée
- Tracking MLflow pour tous les modèles
- Sélection automatique du meilleur modèle
- Sauvegarde du meilleur modèle dans `best_model.txt`

### 3. **experiment_hyper.py** - Optimisation des Hyperparamètres
- Grid Search sur le meilleur modèle identifié
- Validation croisée stratifiée
- Tracking MLflow des hyperparamètres
- Sauvegarde des meilleurs paramètres en JSON

### 4. **model_eval.py** - Évaluation Finale
- Évaluation complète du modèle optimisé
- Métriques détaillées (accuracy, precision, recall, F1, AUC)
- Visualisations (matrice de confusion, courbe ROC)
- Validation croisée sur l'ensemble complet

### 5. **HD_stream.py** - Interface Utilisateur
- Exploration des données (EDA)
- Visualisations interactives
- Test de modèles en temps réel
- Prédiction personnalisée
- Interface utilisateur intuitive

## 🚀 Installation et Utilisation

### Prérequis
```bash
pip install -r requirements.txt
```

### Démarrage de MLflow
```bash
mlflow server --host 127.0.0.1 --port 5000
```

### Exécution du Pipeline

1. **Expérimentation des modèles**:
```bash
python experiment.py
```

2. **Optimisation des hyperparamètres**:
```bash
python experiment_hyper.py
```

3. **Évaluation finale**:
```bash
python model_eval.py
```

4. **Interface utilisateur**:
```bash
streamlit run HD_stream.py
```

## 📋 Dataset

Le dataset `heart-disease.csv` doit contenir les colonnes suivantes:
- `age`: Âge du patient
- `sex`: Sexe (0=Femme, 1=Homme)
- `cp`: Type de douleur thoracique (0-3)
- `trestbps`: Pression artérielle au repos
- `chol`: Cholestérol sérique
- `fbs`: Glycémie à jeun > 120 mg/dl
- `restecg`: Résultats ECG au repos
- `thalach`: Fréquence cardiaque maximale
- `exang`: Angine induite par l'exercice
- `oldpeak`: Dépression ST
- `slope`: Pente du segment ST
- `ca`: Nombre de vaisseaux principaux
- `thal`: Thalassémie
- `target`: Variable cible (0=Pas de maladie, 1=Maladie)

## 🔧 Configuration

### Modèles Testés
- **RandomForest**: Forêt aléatoire
- **GradientBoosting**: Gradient boosting
- **HistGradientBoosting**: Gradient boosting histogramme
- **SVM**: Support Vector Machine
- **LogisticRegression**: Régression logistique

### Hyperparamètres Optimisés
Chaque modèle a sa propre grille de paramètres définie dans `params.yaml`.

## 📈 Tracking MLflow

Le projet utilise MLflow pour:
- Suivre les expériences et leurs paramètres
- Comparer les performances des modèles
- Enregistrer les artefacts (modèles, graphiques)
- Gérer les versions des modèles

Interface accessible sur: http://127.0.0.1:5000

## 🎯 Fonctionnalités de l'Interface

### Exploration des Données
- Statistiques descriptives
- Informations sur les types de données
- Distribution des variables

### Visualisations
- Histogrammes des variables numériques
- Graphiques de comptage pour les variables catégoriques
- Matrice de corrélation
- Analyse bivariée avec la variable cible

### Modélisation
- Test interactif de différents modèles
- Affichage des performances
- Analyse détaillée pour la régression logistique
- Tests statistiques (Khi-deux)

### Prédiction Personnalisée
- Formulaire de saisie des paramètres patient
- Prédiction en temps réel
- Visualisation du risque
- Recommandations basées sur le niveau de risque

## ⚠️ Avertissements

- Cette application est à des fins éducatives uniquement
- Ne remplace pas un avis médical professionnel
- Les prédictions doivent être interprétées par un professionnel de santé

## 🛠️ Améliorations Apportées

### Corrections Techniques
- Gestion des erreurs et exceptions
- Validation des fichiers requis
- Optimisation des performances
- Code plus lisible et maintenable

### Améliorations Fonctionnelles
- Interface utilisateur améliorée
- Visualisations plus riches
- Métriques d'évaluation complètes
- Recommandations médicales contextuelles

### Bonnes Pratiques
- Séparation des responsabilités
- Configuration externalisée
- Documentation complète
- Gestion des versions

## 👤 Auteur

**KABBAJ MOHAMED**
- Développé avec Streamlit, Scikit-learn et MLflow
- Pipeline ML complet pour la prédiction des maladies cardiaques
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

