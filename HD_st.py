# Importations des bibliothèques
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Configuration de la page Streamlit
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("Prédiction des Maladies Cardiaques")

# Chargement des données
@st.cache_data  # Cache pour accélérer le chargement
def load_data():
    df = pd.read_csv('C:/Users/pc/Desktop/KABBAJ DOC/EDUCATIONNEL/data challenge/Heart_disease/heart-disease.csv')
    return df

df = load_data()

# Sidebar pour la navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Choisir une section", ["Exploration des Données", "Visualisations", "Modélisation"])

# Fonction pour le test du Khi-deux
def chi2_test(column, df):
    contingency_table = pd.crosstab(df['target'], df[column])
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2_stat, p_value

# Fonction pour évaluer les modèles
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report, model

# Section 1 : Exploration des Données (EDA)
if section == "Exploration des Données":
    st.header("Exploration des Données (EDA)")

    st.subheader("Aperçu des données")
    st.write(df.head(10))

    st.subheader("Statistiques descriptives")
    st.write(df.describe())

    st.subheader("Informations sur les données")
    buffer = pd.DataFrame(df.dtypes, columns=["Type"])
    buffer["Valeurs manquantes"] = df.isnull().sum()
    st.write(buffer)

# Section 2 : Visualisations
elif section == "Visualisations":
    st.header("Visualisations des Données")
    sns.set(style="darkgrid")

    # Distribution des variables numériques
    st.subheader("Distribution des variables numériques")
    fig, ax = plt.subplots(figsize=(10, 8))
    df.hist(bins=30, edgecolor='black', ax=ax)
    plt.suptitle("Distribution des variables numériques")
    st.pyplot(fig)

    # Fonction pour les countplots
    def plot_countplot(column, title, hue=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.countplot(x=column, data=df, hue=hue)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Nombre')
        st.pyplot(fig)

    # Sélection des visualisations via une liste déroulante
    viz_options = [
        "Sexe", "Type de douleur (cp)", "Pente (slope)", "Thalassémie (thal)", "Cible (target)",
        "Pente par cible", "Thalassémie par cible", "Sexe par cible", "Groupe d'âge par cible", "Pression artérielle par cible"
    ]
    selected_viz = st.selectbox("Choisir une visualisation", viz_options)

    if selected_viz == "Sexe":
        plot_countplot('sex', 'Distribution de la variable sexe')
    elif selected_viz == "Type de douleur (cp)":
        plot_countplot('cp', 'Distribution de la variable cp')
    elif selected_viz == "Pente (slope)":
        plot_countplot('slope', 'Distribution de la variable slope')
    elif selected_viz == "Thalassémie (thal)":
        plot_countplot('thal', 'Distribution de la variable thal')
    elif selected_viz == "Cible (target)":
        plot_countplot('target', 'Distribution de la variable cible (target)')
    elif selected_viz == "Pente par cible":
        plot_countplot('slope', 'Distribution de slope par rapport à la cible', hue='target')
    elif selected_viz == "Thalassémie par cible":
        plot_countplot('thal', 'Distribution de thal par rapport à la cible', hue='target')
    elif selected_viz == "Sexe par cible":
        plot_countplot('sex', 'Distribution du sexe par rapport à la cible', hue='target')
    elif selected_viz == "Groupe d'âge par cible":
        df['age_group'] = pd.cut(df['age'], bins=[25, 40, 45, 50, 55, 60, 65, 80], labels=['25-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-80'])
        plot_countplot('age_group', 'Distribution de l’âge par rapport à la cible', hue='target')
    elif selected_viz == "Pression artérielle par cible":
        df['trestbps_group'] = pd.cut(df['trestbps'], bins=[90, 100, 110, 120, 125, 130, 140, 180, 200], labels=['90-100', '100-110', '110-120', '120-125', '125-130', '130-140', '140-180', '180-200'])
        plot_countplot('trestbps_group', 'Distribution de trestbps par rapport à la cible', hue='target')

    # Matrice de corrélation
    st.subheader("Matrice de corrélation")
    fig, ax = plt.subplots(figsize=(12, 8))
    corr_matrix = df.drop(['age_group', 'trestbps_group'], axis=1, errors='ignore').corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.title('Matrice de corrélation')
    st.pyplot(fig)

# Section 3 : Modélisation
elif section == "Modélisation":
    st.header("Modélisation")

    # Préparation des données
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("Forme des données")
    st.write(f"Données d'entraînement : {X_train.shape}")
    st.write(f"Données de test : {X_test.shape}")
    st.write(f"Étiquettes d'entraînement : {y_train.shape}")
    st.write(f"Étiquettes de test : {y_test.shape}")

    # Sélection du modèle
    model_options = ["Forêt Aléatoire", "Régression Logistique", "Réseau de Neurones", "Arbre de Décision", "SVM"]
    selected_model = st.selectbox("Choisir un modèle", model_options)

    # Initialisation des modèles
    models = {
        "Forêt Aléatoire": RandomForestClassifier(random_state=42),
        "Régression Logistique": LogisticRegression(random_state=42, max_iter=1000),
        "Réseau de Neurones": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
        "Arbre de Décision": DecisionTreeClassifier(random_state=42),
        "SVM": svm.SVC(random_state=42)
    }

    # Évaluation du modèle sélectionné
    if st.button("Évaluer le modèle"):
        model = models[selected_model]
        accuracy, report, trained_model = evaluate_model(model, X_train, X_test, y_train, y_test, selected_model)
        st.subheader(f"Résultats pour {selected_model}")
        st.write(f"**Précision** : {accuracy:.4f}")
        st.write("**Rapport de classification** :")
        st.write(pd.DataFrame(report).T)

        # Résultats détaillés pour la régression logistique
        if selected_model == "Régression Logistique":
            X_train_sm = sm.add_constant(X_train)
            logit_model = sm.Logit(y_train, X_train_sm)
            result = logit_model.fit()
            st.subheader("Résultats détaillés de la régression logistique")
            st.text(result.summary().as_text())

            odds_ratios = np.exp(result.params)
            conf_int = np.exp(result.conf_int())
            conf_int.columns = ['2.5%', '97.5%']
            st.write("**Odds Ratios** :")
            st.write(pd.DataFrame(odds_ratios, columns=["Odds Ratio"]))
            st.write("**Intervalles de confiance des Odds Ratios** :")
            st.write(conf_int)

    # Test du Khi-deux pour les variables catégoriques
    st.subheader("Test du Khi-deux")
    categorical_vars = ['sex', 'cp', 'slope', 'thal']
    for var in categorical_vars:
        chi2_stat, p_value = chi2_test(var, df)
        st.write(f"**{var}** : Statistique = {chi2_stat:.4f}, p-value = {p_value:.4f}")
        if p_value < 0.05:
            st.write(f"{var} est significativement associée aux maladies cardiaques.")
        else:
            st.write(f"Aucune association significative entre {var} et les maladies cardiaques.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Développé avec Streamlit par [KABBAJ MOHAMED]")