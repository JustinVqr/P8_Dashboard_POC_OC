import streamlit as st
import pandas as pd
import requests
from io import StringIO
import os
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configuration de la page
st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title="Accueil")

# Menu de navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisissez une page", ["Accueil", "Données", "Analyse des clients", "Prédiction"])

# --- Initialisation de l'état de session ---
if "load_state" not in st.session_state:
    st.session_state.load_state = False

# --- Fonction pour télécharger et charger les données ---
@st.cache_data
def load_data():
    url_train = "https://www.dropbox.com/scl/fi/9oc8a12r2pnanhzj2r6gu/df_train.csv?rlkey=zdcao9gluupqkd3ljxwnm1pv6&st=mm5480h6&dl=1"
    url_new = "https://www.dropbox.com/scl/fi/2mylh9bshf5jkzg6n9m7t/df_new.csv?rlkey=m82n87j6hr9en1utkt7a8qsv4&st=k6kj1pm5&dl=1"
    
    response_train = requests.get(url_train)
    response_new = requests.get(url_new)
    
    if response_train.status_code == 200 and response_new.status_code == 200:
        df_train = pd.read_csv(StringIO(response_train.text), sep=',', index_col="SK_ID_CURR", encoding='utf-8')
        df_new = pd.read_csv(StringIO(response_new.text), sep=',', index_col="SK_ID_CURR", encoding='utf-8')
        return df_train, df_new
    else:
        st.error(f"Erreur de téléchargement : Statut {response_train.status_code}, {response_new.status_code}")
        return None, None

# --- Fonction pour échantillonnage stratifié ---
def stratified_sampling(df, target_column='TARGET', sample_size=0.1):
    df_sampled, _ = train_test_split(df, test_size=1-sample_size, stratify=df[target_column], random_state=42)
    return df_sampled

# --- Chargement du modèle et de l'explainer ---
def load_model_and_explainer(df_train):
    model_path = os.path.join(os.getcwd(), 'app', 'model', 'best_model.pkl')
    if os.path.exists(model_path):
        try:
            Credit_clf_final = joblib.load(model_path)
            st.write("Modèle chargé avec succès.")
            try:
                explainer = shap.TreeExplainer(Credit_clf_final, df_train.drop(columns="TARGET").fillna(0))
            except Exception as e:
                st.write(f"TreeExplainer non compatible : {e}. Utilisation de KernelExplainer.")
                explainer = shap.KernelExplainer(Credit_clf_final.predict, df_train.drop(columns="TARGET").fillna(0))
            return Credit_clf_final, explainer
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle ou de l'explicateur : {e}")
            return None, None
    else:
        st.error(f"Le fichier {model_path} n'existe pas.")
        return None, None

# --- Fonction pour afficher la page d'accueil ---
def show_home_page():
    st.title("Accueil")
    st.write("Bienvenue dans l'application d'aide à la décision de prêt.")

# --- Fonction pour afficher la page d'analyse des clients ---
def show_analysis_page():
    st.title("Analyse des clients")
    # Code pour l'analyse des clients

# --- Fonction pour afficher la page de prédiction ---
def show_prediction_page():
    st.title("Prédiction")
    sk_id_curr = st.text_input("Entrez l'ID du client pour obtenir la prédiction :")
    if st.button("Obtenir la prédiction"):
        if sk_id_curr and st.session_state.get("Credit_clf_final") and st.session_state.get("explainer"):
            try:
                client_id = int(sk_id_curr)
                if client_id in st.session_state.df_new.index:
                    df_client = st.session_state.df_new.loc[[client_id]]
                    X_client = df_client.fillna(0)
                    prediction_proba = st.session_state.Credit_clf_final.predict_proba(X_client)[:, 1]
                    prediction = st.session_state.Credit_clf_final.predict(X_client)
                    st.write(f"Prédiction : {'Oui' if prediction[0] == 1 else 'Non'}")
                    st.write(f"Probabilité de défaut : {prediction_proba[0] * 100:.2f}%")
                    shap_values = st.session_state.explainer.shap_values(X_client)
                    st.write("Valeurs SHAP calculées.")
                    shap.initjs()
                    expected_value = st.session_state.explainer.expected_value[1] if isinstance(st.session_state.explainer.expected_value, list) else st.session_state.explainer.expected_value
                    st.pyplot(shap.force_plot(expected_value, shap_values[1][0], X_client, matplotlib=True))
                else:
                    st.error("Client ID non trouvé.")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
        else:
            st.error("Modèle non chargé ou ID client invalide.")

# --- Logique de la page sélectionnée ---
if not st.session_state.get("load_state"):
    df_train, df_new = load_data()
    if df_train is not None and df_new is not None:
        df_train_sampled = stratified_sampling(df_train, sample_size=0.1)
        Credit_clf_final, explainer = load_model_and_explainer(df_train_sampled)
        if Credit_clf_final and explainer:
            st.session_state.Credit_clf_final = Credit_clf_final
            st.session_state.explainer = explainer
            st.session_state.df_train = df_train_sampled
            st.session_state.df_new = df_new
            st.session_state.load_state = True
            st.success("Modèle et explicateur SHAP chargés avec succès.")
else:
    df_train = st.session_state.df_train
    df_new = st.session_state.df_new

# Sélection de la page à afficher
if page == "Accueil":
    show_home_page()
elif page == "Analyse des clients":
    show_analysis_page()
elif page == "Prédiction":
    show_prediction_page()
