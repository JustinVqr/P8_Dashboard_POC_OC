import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(
    layout='centered',
    initial_sidebar_state='collapsed',
    page_title="2) Analyse clients connus"
)

# Ajoutez le chemin du répertoire racine au sys.path pour que Python trouve les modules dans 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.component.P7_App_FrontEnd import execute_noAPI, plot_client, nan_values, shap_plot, plot_gauge

# Vérification que les données sont disponibles dans le session state
if 'df_train' not in st.session_state or 'Credit_clf_final' not in st.session_state or 'explainer' not in st.session_state:
    st.error("Les données nécessaires ne sont pas disponibles dans l'état de session. Veuillez charger les données sur la page d'accueil.")
    st.stop()

# Chargement des données depuis l'état de session
df_train = st.session_state.df_train
Credit_clf_final = st.session_state.Credit_clf_final
explainer = st.session_state.explainer

# Affichage de l'en-tête principal
st.header("Analyse du défaut de paiement des clients connus")

# Explication de la section pour l'utilisateur
st.markdown("""
    **Instructions :**  
    Utilisez cette page pour entrer l'ID d'un client connu. Le modèle prédira la probabilité de défaut de paiement 
    et affichera les graphiques explicatifs pour mieux comprendre le profil du client.
""")

# Boîte de saisie pour l'ID du client (centrée sur la page)
index_client = st.number_input(
    "Entrez l'ID du client (ex : 1, 2, 3, 5)",
    format="%d",
    value=1
)

# Bouton d'exécution (centré sur la page)
run_btn = st.button('Voir les données du client')

# Action déclenchée par le bouton
if run_btn:
    # Vérification de la présence de l'ID dans df_train
    if index_client in df_train.index:
        try:
            # Appel de la fonction principale pour traiter les données du client
            execute_noAPI(df_train, index_client, Credit_clf_final)

            # Calculer la probabilité de défaut de paiement pour le client
            X_client = df_train.loc[[index_client]].drop(columns='TARGET').fillna(0)
            pred_prob = Credit_clf_final.predict_proba(X_client)[0][1]

            # Affichage de la jauge avant les graphiques SHAP
            plot_gauge(pred_prob)

            # Commentaire sur la jauge
            st.markdown("""
            **Interprétation de la jauge :**  
            Cette jauge représente la probabilité de défaut de paiement du client. Une probabilité proche de 1 indique un risque élevé, tandis qu'une valeur proche de 0 indique un risque faible.
            """)

            # --- Volet rétractable pour les graphiques SHAP ---
            with st.expander("Voir l'explication de la prédiction pour le client"):
                shap_plot(explainer, df_train.drop(columns='TARGET').fillna(0), index_client=index_client)
                st.markdown("""
                **Analyse des graphiques SHAP :**  
                Ces graphiques SHAP montrent les variables les plus influentes sur la décision du modèle. Pour le premier graphique les barres positives, en rouge, augmentent la probabilité de défaut de paiement, et les barres négatives, en bleu, la réduisent. Le second graphique quand à lui donne une lecture globale, où nous voyons la contribution des features que ce soit pour réduire ou augmenter le score-crédit obtenu.
                """)

            # --- Organisation des autres graphiques dans une colonne ---
            plot_client(
                df_train.drop(columns='TARGET').fillna(0),  # Gestion des NaN
                explainer,
                df_reference=df_train,
                index_client=index_client
            )
            st.markdown("""
            **Analyse du profil du client :**  
            Le graphique ci-dessus compare les caractéristiques du client à celles d'autres clients dans la base de données. Cela permet de comprendre où se situe le client par rapport à la population.
            """)

            # --- Affichage du message des valeurs manquantes en dessous des graphiques ---
            nan_values(df_train.drop(columns='TARGET'), index_client=index_client)
            st.markdown("""
            **Valeurs manquantes :**  
            Si certaines données du client sont manquantes, elles seront listées ici. Cela peut avoir un impact sur la précision des prédictions.
            """)

        except Exception as e:
            st.error(f"Une erreur s'est produite lors de l'affichage des données du client : {e}")
    else:
        st.error("Client non présent dans la base de données")
