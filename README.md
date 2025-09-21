# Projet de Prédiction des Maladies Cardiaques

Ce projet implémente un pipeline complet de machine learning pour prédire les maladies cardiaques en utilisant des données médicales. Il comprend l'expérimentation de modèles, l'optimisation d'hyperparamètres, l'évaluation, une interface utilisateur interactive et un déploiement automatisé avec Docker et CI/CD.

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
├── Dockerfile                 # Configuration Docker
├── .github/workflows/
│   └── ci.yaml               # Pipeline CI/CD
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

1. **Initialiser DVC**:
```bash
dvc init
dvc pull
```

2. **Exécuter le pipeline**:
```bash
dvc repro 
```

3. **Interface utilisateur**:
```bash
streamlit run HD_stream.py
```

## 🐳 Déploiement avec Docker

### Utilisation Locale

1. **Construire l'image Docker**:
```bash
docker build -t heart-disease-app .
```

2. **Lancer le conteneur**:
```bash
docker run -p 8080:8080 heart-disease-app
```

3. **Accéder à l'application**:
```
http://localhost:8080
```

### Utilisation avec l'image Docker Hub

```bash
# Télécharger l'image depuis Docker Hub
docker pull kabbajstat/my-ml-pipeline:latest

# Lancer le conteneur
docker run -p 8080:8080 kabbajstat/my-ml-pipeline:latest
```

## 🔄 Pipeline CI/CD

### Déploiement Automatique

Le projet utilise GitHub Actions pour un déploiement automatique :

1. **Déclenchement** : Push sur la branche `master`
2. **Tests** : Exécution du pipeline DVC
3. **Build** : Construction de l'image Docker
4. **Deploy** : Publication sur Docker Hub

### Configuration des Secrets

Pour le déploiement automatique, configurez ces secrets dans votre repository GitHub :

```
DOCKER_USERNAME = votre_nom_utilisateur_docker_hub
DOCKER_PASSWORD = votre_mot_de_passe_ou_token_docker_hub
```

### Workflow CI/CD

Le pipeline automatique :
- ✅ Installe les dépendances Python
- ✅ Exécute le pipeline DVC
- ✅ Construit l'image Docker
- ✅ Pousse l'image vers Docker Hub
- ✅ Déploie automatiquement l'application

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

## 🛠️ Technologies Utilisées

### Machine Learning & Data Science
- **Scikit-learn** : Modèles de machine learning
- **Pandas** : Manipulation des données
- **NumPy** : Calculs numériques
- **MLflow** : Tracking des expériences

### Interface & Visualisation
- **Streamlit** : Interface utilisateur web
- **Matplotlib** : Visualisations
- **Seaborn** : Visualisations statistiques

### DevOps & Déploiement
- **Docker** : Containerisation
- **GitHub Actions** : CI/CD
- **Docker Hub** : Registry d'images
- **DVC** : Versioning des données et pipelines

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

### DevOps & Déploiement
- **Containerisation** avec Docker
- **CI/CD automatisé** avec GitHub Actions
- **Déploiement sur Docker Hub**
- **Pipeline reproductible** avec DVC

### Bonnes Pratiques
- Séparation des responsabilités
- Configuration externalisée
- Documentation complète
- Gestion des versions
- Tests automatisés

## 🚀 Démarrage Rapide

### Option 1 : Utilisation Docker (Recommandée)
```bash
docker run -p 8080:8080 kabbajstat/my-ml-pipeline:latest
```
Accédez à http://localhost:8080

### Option 2 : Installation Locale
```bash
git clone https://github.com/kabbstat/Heart_disease.git
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run HD_stream.py
```

### Option 3 : Développement
```bash
git clone https://github.com/kabbstat/Heart_disease.git
cd heart-disease-prediction
pip install -r requirements.txt
mlflow server --host 127.0.0.1 --port 5000
dvc repro
streamlit run HD_stream.py
```

## 👤 Auteur

**KABBAJ MOHAMED**
- Développé avec Streamlit, Scikit-learn et MLflow
- Pipeline ML complet pour la prédiction des maladies cardiaques
- Déploiement automatisé avec Docker et CI/CD
