# Importations des bibliothèques
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

# Chargement des données
df = pd.read_csv('C:/Users/pc/Desktop/KABBAJ DOC/EDUCATIONNEL/data challenge/Heart_disease/heart-disease.csv')

# Exploration des données (EDA)
print("=== Aperçu des données ===")
print(df.head(10))
print("\n=== Statistiques descriptives ===")
print(df.describe())
print("\n=== Informations sur les données ===")
print(df.info())

# Visualisation des données (Data Visualization)
sns.set(style="darkgrid")

# Distribution des variables numériques
plt.figure(figsize=(10, 8))
df.hist(figsize=(10, 8), bins=30, edgecolor='black')
plt.suptitle("Distribution des variables numériques")
plt.show()

# Distribution des variables catégorielles
def plot_countplot(column, title, hue=None):
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=column, data=df, hue=hue)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()))
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

plot_countplot('sex', 'Distribution de la variable sexe')
plot_countplot('cp', 'Distribution de la variable cp')
plot_countplot('slope', 'Distribution de la variable slope')
plot_countplot('thal', 'Distribution de la variable thal')
plot_countplot('target', 'Distribution de la variable cible (target)')

# Distribution de la variable cible par rapport à d'autres variables
plot_countplot('slope', 'Distribution de slope par rapport à la cible', hue='target')
plot_countplot('thal', 'Distribution de thal par rapport à la cible', hue='target')

# Distribution de l'âge par rapport à la cible
df['age_group'] = pd.cut(df['age'], bins=[25, 40, 45, 50, 55, 60, 65, 80], labels=['25-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-80'])
plot_countplot('age_group', 'Distribution de l\'âge par rapport à la cible', hue='target')

# Distribution de trestbps par rapport à la cible
df['trestbps_group'] = pd.cut(df['trestbps'], bins=[90, 100, 110, 120, 125, 130, 140, 180, 200], labels=['90-100', '100-110', '110-120', '120-125', '125-130', '130-140', '140-180', '180-200'])
plot_countplot('trestbps_group', 'Distribution de trestbps par rapport à la cible', hue='target')

# Matrice de corrélation
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation')
plt.show()

# Test du Khi-deux pour les variables catégorielles
def chi2_test(column):
    contingency_table = pd.crosstab(df['target'], df[column])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Test du Khi-deux pour {column}:")
    print(f"Statistique de test: {chi2_stat:.4f}, p-value: {p_value:.4f}")

chi2_test('sex')
chi2_test('cp')
chi2_test('slope')
chi2_test('thal')

# Préparation des données pour le modèle
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== Forme des données ===")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

# Modélisation

## Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n=== Random Forest ===")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

## Régression Logistique
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred)
print("\n=== Régression Logistique ===")
print("Accuracy:", accuracy_lr)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Coefficients et p-values de la régression logistique
X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()
print("\n=== Résultats détaillés de la régression logistique ===")
print(result.summary())

# Odds Ratios
odds_ratios = np.exp(result.params)
print("\n=== Odds Ratios ===")
print(odds_ratios)

# Intervalles de confiance des Odds Ratios
conf_int = result.conf_int()
conf_int.columns = ['2.5%', '97.5%']
conf_int_odds = np.exp(conf_int)
print("\n=== Intervalles de confiance des Odds Ratios ===")
print(conf_int_odds)

## Réseau de neurones (ANN)
ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
ann.fit(X_train, y_train)
y_pred = ann.predict(X_test)
accuracy_ann = accuracy_score(y_test, y_pred)
print("\n=== Réseau de Neurones (ANN) ===")
print("Accuracy:", accuracy_ann)
print("Classification Report:\n", classification_report(y_test, y_pred))

## Arbre de décision
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred)
print("\n=== Arbre de Décision ===")
print("Accuracy:", accuracy_dt)
print("Classification Report:\n", classification_report(y_test, y_pred))

## SVM
sup_vm = svm.SVC(random_state=42)
sup_vm.fit(X_train, y_train)
y_pred = sup_vm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred)
print("\n=== SVM ===")
print("Accuracy:", accuracy_svm)
print("Classification Report:\n", classification_report(y_test, y_pred))