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
st.set_page_config(layout='wide', page_title="Accueil")

# --- Initialisation de l'état de session ---
if "load_state" not in st.session_state:
    st.session_state.load_state = False

# --- Fonction pour télécharger et charger les données ---
@st.cache_data
def load_data():
    url_train = "https://www.dropbox.com/scl/fi/tupb249wgfu00bobae1ij/df_train.csv?rlkey=ce5jh618t0st7f5isuusy6mc6&st=ahj8us3v&dl=1"
    url_new = "https://www.dropbox.com/scl/fi/y2hschethh5vzy52acwk9/df_new.csv?rlkey=u536gklarq459hrrak9mjdexx&st=x0cdjshe&dl=1"
    
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

# --- Chargement du modèle et initialisation de l'explicateur SHAP ---
def load_model():
    model_path = os.path.join(os.getcwd(), 'app', 'model', 'best_model.pkl')
    if os.path.exists(model_path):
        try:
            Credit_clf_final = joblib.load(model_path)
            # Initialisation de l'explicateur SHAP avec le modèle chargé
            explainer = shap.TreeExplainer(Credit_clf_final)
            st.session_state.Credit_clf_final = Credit_clf_final
            st.session_state.explainer = explainer
            st.write("Modèle et explainer SHAP chargés avec succès.")
            return Credit_clf_final
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")
            return None
    else:
        st.error(f"Le fichier {model_path} n'existe pas.")
        return None

# --- Logique de chargement initial ---
if not st.session_state.load_state:
    df_train, df_new = load_data()
    if df_train is not None and df_new is not None:
        df_train_sampled = stratified_sampling(df_train, sample_size=0.1)
        Credit_clf_final = load_model()
        if Credit_clf_final:
            st.session_state.df_train = df_train_sampled
            st.session_state.df_new = df_new
            st.session_state.load_state = True
            # Afficher "Application opérationnelle" sous l'image, centré
            st.markdown("<h4 style='text-align: center;'>Application opérationnelle</h4>", unsafe_allow_html=True)
else:
    df_train = st.session_state.df_train
    df_new = st.session_state.df_new

# --- Fonction pour afficher la page d'accueil ---
def show_home_page():
    st.title("Accueil")
    st.write("Bienvenue dans l'application d'aide à la décision de prêt.")
    st.write("""Cette application aide l'agent de prêt dans sa décision d'accorder un prêt à un client.
     Pour ce faire, un algorithme de machine learning est utilisé pour prédire les difficultés d'un client à rembourser le prêt.
     Pour plus de transparence, celle-ci fournit également des informations pour expliquer l'algorithme et les prédictions, selon les caractéristiques du client étudié.""")
    
    # --- Logo ---
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://raw.githubusercontent.com/JustinVqr/P7_ScoringModel/main/app/images/logo.png")
    # Afficher le message "Application opérationnelle"
#    st.markdown("<h4 style='text-align: center;'>Application opérationnelle</h4>", unsafe_allow_html=True)

# --- Fonction pour afficher la page d'analyse des clients ---
def show_analysis_page():
    st.title("Analyse des clients")
    col1, col2 = st.columns(2)
    # Code pour l'analyse des clients

# --- Fonction pour afficher la page de prédiction ---
def show_prediction_page():
    st.title("Prédiction")
    sk_id_curr = st.text_input("Entrez l'ID du client pour obtenir la prédiction :")
    if st.button("Obtenir la prédiction"):
        if sk_id_curr and st.session_state.get("Credit_clf_final"):
            try:
                client_id = int(sk_id_curr)
                if client_id in st.session_state.df_new.index:
                    df_client = st.session_state.df_new.loc[[client_id]]
                    X_client = df_client.fillna(0)
                    # Prédiction du modèle
                    prediction_proba = st.session_state.Credit_clf_final.predict_proba(X_client)[:, 1]
                    prediction = st.session_state.Credit_clf_final.predict(X_client)
                    st.write(f"Prédiction : {'Oui' if prediction[0] == 1 else 'Non'}")
                    st.write(f"Probabilité de défaut : {prediction_proba[0] * 100:.2f}%")
                    
                    # Utilisation de l'explicateur SHAP stocké dans st.session_state
                    explainer = st.session_state.explainer
                    shap_values_client = explainer.shap_values(X_client)
                    
                    # Vérification du type de shap_values_client
                    if isinstance(shap_values_client, list):
                        shap_values_client = shap_values_client[1]
                    
                    # Visualisation des valeurs SHAP pour le client
                    shap.initjs()
                    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                    st.pyplot(shap.force_plot(expected_value, shap_values_client[0], X_client, matplotlib=True))
                else:
                    st.error("Client ID non trouvé.")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
        else:
            st.error("Modèle non chargé ou ID client invalide.")

# Sélection de la page à afficher
query_params = st.query_params 
page = query_params.get("page", ["Accueil"])[0]

if page == "Accueil":
    show_home_page()
elif page == "Analyse des clients":
    show_analysis_page()
elif page == "Prédiction":
    show_prediction_page()