import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import sys
import os

# Configuration de la page Streamlit
st.set_page_config(
    layout='centered',
    initial_sidebar_state='collapsed',
    page_title="1) Présentation des données"
)

# Ajouter le chemin du répertoire racine au sys.path pour que Python trouve les modules dans 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importation des fonctions
from app.component.P7_App_FrontEnd import scatter_plot_interactif, univariate_analysis, radar_plot

# --- Vérification et récupération des données depuis la session ---
if "df_train" not in st.session_state or "Credit_clf_final" not in st.session_state or "explainer" not in st.session_state:
    st.error("Les données nécessaires ne sont pas disponibles dans la session. Veuillez vous assurer que le modèle et les données sont chargés depuis la page d'accueil.")
    st.stop()

# Chargement des données depuis l'état de session
df_train = st.session_state.df_train
Credit_clf_final = st.session_state.Credit_clf_final
explainer = st.session_state.explainer

# --- Création de la mise en page de la page (3 onglets) ---
tab1, tab2, tab3 = st.tabs(["Dataset", "Analyse", "Modèle"])

# --- Onglet 1 : Présentation du dataframe ---
with tab1:
    st.header("Aperçu du Dataset")
    st.subheader("Informations générales")
    st.write("""
        Cet onglet vous permet de consulter un aperçu du dataset utilisé pour l'entraînement du modèle.
        Vous pouvez voir ici le nombre total de clients enregistrés ainsi que les caractéristiques disponibles pour chaque client.
    """)
    col1, col2 = st.columns(2)
    col1.metric("Nombre de clients enregistrés", df_train.shape[0])
    col2.metric("Nombre de caractéristiques des clients", df_train.drop(columns='TARGET').shape[1])
    
    # Analyse de la cible : Diagramme en anneau
    st.subheader("Répartition de la cible")
    fig1, ax = plt.subplots()
    ax.pie(df_train.TARGET.value_counts(normalize=True),
           labels=["0", "1"],
           autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, (p / 100) * sum(df_train.TARGET.value_counts())),
           startangle=0,
           pctdistance=0.8,
           explode=(0.05, 0.05))
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    plt.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.title('Répartition des clients ayant des difficultés (1) ou non (0) à rembourser le prêt')
    plt.tight_layout()
    st.pyplot(fig1)

    # Analyse des valeurs manquantes
    st.subheader("Analyse des valeurs manquantes")
    figNAN = msno.matrix(df_train.drop(columns='TARGET'), labels=True, sort="ascending")
    st.pyplot(figNAN.figure)

# --- Onglet 2 : Analyse des caractéristiques ---
with tab2:
    st.header("Analyse des caractéristiques avec univariée, bivariée et radar plot")
    st.write("""
        Explorez les distributions univariées, les relations bivariées entre variables, et comparez les caractéristiques d'un client spécifique avec la moyenne des clients via le radar plot. Sélectionnez les graphiques ci-dessous pour plus de détails.
    """)

    # Ajout du radar plot pour la comparaison client vs moyenne
    with st.expander("Comparaison radar plot : Client vs Moyenne des clients"):
        st.write("""
            Le radar plot ci-dessous compare les caractéristiques d'un client spécifique avec la moyenne de tous les clients. Sélectionnez un ID client pour voir la comparaison.
        """)
        client_id = st.selectbox("Sélectionnez un ID client", df_train.index)
        radar_plot(df_train, client_id, df_train.columns.drop(['TARGET']))

    # Ajout de l'analyse univariée
    with st.expander("Analyse univariée : distribution des variables"):
        st.write("""
            Explorez la distribution individuelle des variables. Ce graphique vous aide à comprendre la répartition des différentes caractéristiques des clients.
        """)
        univariate_analysis(df_train)

    # Ajout du scatter plot interactif
    with st.expander("Analyse bivariée : nuage de points"):
        st.write("""
            Le nuage de points interactif ci-dessous montre la relation entre deux variables. Sélectionnez les variables à comparer pour visualiser leur interaction.
        """)
        scatter_plot_interactif(df_train)

# --- Onglet 3 : Présentation du modèle ---
with tab3:
    st.header("Présentation du modèle LightGBM")
    st.write("""
        Cet onglet présente le modèle utilisé pour la classification des clients. Vous y trouverez les caractéristiques importantes, les paramètres optimisés, ainsi que les scores obtenus lors de la validation croisée.
    """)
    # Importance des caractéristiques
    st.subheader("Importance des caractéristiques du modèle LightGBM")
    st.image("https://raw.githubusercontent.com/JustinVqr/P7_ScoringModel/main/app/images/Plot_importance.png")
    # Paramètres optimisés du modèle
    st.subheader("Paramètres (optimisés avec Optuna)")
    if hasattr(Credit_clf_final, 'get_params'):
        st.table(pd.DataFrame.from_dict(Credit_clf_final.get_params(), orient='index', columns=['Paramètre']))
    else:
        st.warning("Les paramètres du modèle ne sont pas disponibles.")
    # Scores obtenus par validation croisée
    st.subheader("Scores obtenus par validation croisée")
    st.write(pd.DataFrame({
        'Metrics': ["AUC", "Accuracy", "F1", "Precision", "Recall", "Profit"],
        'Valeur': [0.764, 0.869, 0.311, 0.271, 0.366, 1.928],
    }))
    # Matrice de confusion et courbe ROC sur le jeu de test
    st.subheader("Courbe ROC et matrice de confusion sur un jeu de test")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://raw.githubusercontent.com/JustinVqr/P7_ScoringModel/main/app/images/Test_ROC_AUC.png")
    with col2:
        st.image("https://raw.githubusercontent.com/JustinVqr/P7_ScoringModel/main/app/images/Test_confusion_matrix.png")
