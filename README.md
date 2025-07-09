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

