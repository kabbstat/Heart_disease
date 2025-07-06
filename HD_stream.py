# Importations des bibliothèques
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_validate
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
from utils import load_data, split_data, get_model_class, load_best_model, load_best_params

# Configuration de la page Streamlit
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
col1, col2, col3 = st.columns([1,2,1])
with col2:
  st.title("Prédiction des Maladies Cardiaques")
with col2:
  st.image("Heart-Disease.jpg", caption="Prédiction des Maladies Cardiaques", width=500)

# Chargement des données
@st.cache_data  # Cache pour accélérer le chargement
def load_data():
    df = pd.read_csv('heart-disease.csv')
    return df

df = load_data()

# Sidebar pour la navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Choisir une section", ["Exploration des Données", "Visualisations", "Modélisation", "Prédiction personnalisé"])

# Fonction pour le test du Khi-deux
def chi2_test(column, df):
    contingency_table = pd.crosstab(df['target'], df[column])
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2_stat, p_value

# Fonction pour évaluer les modèles
def cross_vali(model, X, y, cv):
    cv = cross_validate(model, X, y, cv=cv,scoring='accuracy', n_jobs=-1)
    accuracy_score_mean = cv['test_score'].mean()
    accuracy_score_st = cv['test_score'].mean()
    best_params_cv = cv
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
    buffer["Valeurs manquantes"] = df.isna().sum()
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
        "Pente vs la cible", "Thalassémie vs la cible", "Sexe vs la cible", "Groupe d'âge vs la cible", "Pression artérielle vs la cible"
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
    elif selected_viz == "Pente vs la cible":
        plot_countplot('slope', 'Distribution de slope par rapport à la cible', hue='target')
    elif selected_viz == "Thalassémie vs la cible":
        plot_countplot('thal', 'Distribution de thal par rapport à la cible', hue='target')
    elif selected_viz == "Sexe vs la cible":
        plot_countplot('sex', 'Distribution du sexe par rapport à la cible', hue='target')
    elif selected_viz == "Groupe d'âge vs la cible":
        df['age_group'] = pd.cut(df['age'], bins=[25, 40, 45, 50, 55, 60, 65, 80], labels=['25-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-80'])
        plot_countplot('age_group', 'Distribution de l’âge par rapport à la cible', hue='target')
    elif selected_viz == "Pression artérielle vs la cible":
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

    # Préparation des données et du modele 
    X,y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    best_model_name = load_best_model()
    best_params = load_best_params()
    
    
    st.subheader("Forme des données")
    st.write(f"Données d'entraînement : {X_train.shape}")
    st.write(f"Données de test : {X_test.shape}")
    st.write(f"Étiquettes d'entraînement : {y_train.shape}")
    st.write(f"Étiquettes de test : {y_test.shape}")
    st.subheader("Experimentation avec mlflow")
    st.write(f"le meilleure model est {best_model_name}")
    st.write(f"best hyperparemeter pour ce modele avec un grid_search est {best_params}")
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

# Prédiction personnalisé
elif section == "Prédiction personnalisé":
    st.header("Prédiction personnalisé")
    st.write("Entrez les valeurs pour chaque variable afin d'estimer la probabilité de subir la crise cardiaque")
    st.write("Le modèle utiliser pour la prédiction est la regression logistique.")
    # entrainement du modèle avec la regression logistique (le modèle le plus performant)
    X=df.drop('target', axis=1)
    y=df["target"]
    default_model = LogisticRegression(random_state=42, max_iter=1000)
    default_model.fit(X,y)
    # formulaire pour sasir les valeurs
    with st.form(key='prediction_form'):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge (years)", min_value=0, max_value=120, value=50)
            sex = st.selectbox("Sexe", options=[0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
            cp = st.selectbox("Type de douleur thoracique", options=[0, 1, 2, 3], format_func=lambda x: ["Angine typique", "Angine atypique", "Non angineuse", "Asymptomatique"][x])
            trestbps = st.number_input("Pression artérielle au repos (mm Hg)", min_value=50, max_value=250, value=120)
            chol = st.number_input("Cholestérol sérique (mg/dl)", min_value=50, max_value=600, value=200)
            fbs = st.selectbox("Glycémie à jeun > 120 mg/dl", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
            restecg = st.selectbox("Résultats ECG au repos", options=[0, 1, 2], format_func=lambda x: ["Normal", "Anomalie ST-T", "Hypertrophie"][x])
        with col2:
            thalach = st.number_input("Fréquence cardiaque max", min_value=50, max_value=250, value=150)
            exang = st.selectbox("Angine à l'effort", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
            oldpeak = st.number_input("Dépression ST (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Pente ST", options=[0, 1, 2], format_func=lambda x: ["Ascendante", "Plate", "Descendante"][x])
            ca = st.number_input("Nombre de vaisseaux (0-3)", min_value=0, max_value=4, value=0)
            thal = st.selectbox("Thalassémie", options=[0, 1, 2, 3], format_func=lambda x: ["Non spécifié", "Normal", "Défaut fixe", "Défaut réversible"][x])

        submit_button = st.form_submit_button(label="Prédire")
    if submit_button: # prédire si le button est cliqué
        # création d'une dataframe avec les valeurs saisies
        input_data= pd.DataFrame({'age':[age],'sex':[sex], 'cp':[cp], 'trestbps':[trestbps], 'chol':[chol], 'fbs':[fbs], 'restecg':[restecg],
                                  'thalach':[thalach],'exang':[exang], 'oldpeak':[oldpeak], 'slope':[slope],'ca':[ca],'thal':[thal]}) 
        proba = default_model.predict_proba(input_data)[0]
        proba_log = default_model.predict_log_proba(input_data)
        proba_heart_disease = proba[1] * 100  
        st.subheader("Résultat de la prédiction")
        st.write(proba_log)
        st.write(f"probabilité de maladie cardiaque: {proba_heart_disease:.2f}%")
        if proba_heart_disease > 50:
            st.warning("Risque élevé de maladie cardiaque detecté")
        else: st.success("Risque faible de maladie cardiaque.")
# Footer
st.sidebar.markdown("---")
st.sidebar.write("Développé avec Streamlit par [KABBAJ MOHAMED]")