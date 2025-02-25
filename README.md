## HEART DISEASE PROJECT 
This repository contains a machine learning project designed to predict the likelihood of heart disease in patients using clinical data. The objective is to develop a supervised learning model capable of classifying patients as either at risk (heart disease present) or not at risk (no heart disease), leveraging various medical attributes.
# Project Overview
Heart disease remains a leading cause of mortality worldwide, making early detection critical. This project applies machine learning techniques to analyze patient data and provide predictive insights, potentially aiding healthcare professionals in decision-making. The dataset includes 303 patient records with 14 attributes, and the final model aims to achieve high accuracy and interpretability.
# Dataset
The dataset is sourced from the UCI Machine Learning Repository and comprises 303 entries with the following 14 features:

Age: Patient's age in years (e.g., 29–77).
Sex: Gender (1 = Male, 0 = Female).
CP: Chest pain type (0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic).
Trestbps: Resting blood pressure in mm Hg (e.g., 94–200).
Chol: Serum cholesterol in mg/dl (e.g., 126–564).
Fbs: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False).
Restecg: Resting electrocardiographic results (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy).
Thalach: Maximum heart rate achieved (e.g., 71–202).
Exang: Exercise-induced angina (1 = Yes, 0 = No).
Oldpeak: ST depression induced by exercise relative to rest (e.g., 0–6.2).
Slope: Slope of the peak exercise ST segment (0 = Upsloping, 1 = Flat, 2 = Downsloping).
Ca: Number of major vessels (0–4) colored by fluoroscopy.
Thal: Thalassemia status (0 = Not specified, 1 = Normal, 2 = Fixed defect, 3 = Reversible defect).
Target: Presence of heart disease (1 = Heart disease, 0 = No heart disease).
The dataset provided includes real-world clinical measurements.
# Models Used
Several machine learning algorithms were tested for this project:

Logistic Regression: A baseline linear model for binary classification.
Decision Tree: A tree-based model for interpretability and feature importance analysis.
Random Forest: An ensemble method to improve accuracy and reduce overfitting.
Support Vector Machine (SVM): A robust classifier for handling non-linear relationships.

Each model was evaluated using metrics such as accuracy, precision, recall, and F1-score, with hyperparameter tuning applied where applicable.
