import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import shap
import json

# Configuration de la page Streamlit
st.set_page_config(
    layout='centered',
    initial_sidebar_state='collapsed',
    page_title="3) Prédictions sur de nouveaux clients"
)

# Ajoutez le chemin du répertoire racine au sys.path pour que Python trouve les modules dans 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.component.P7_App_FrontEnd import execute_API, plot_client, shap_plot, nan_values, plot_gauge

# --- Récupération des données depuis la page d'accueil ---
df_train = st.session_state.df_train  # DataFrame principal contenant les données d'entraînement
df_new = st.session_state.df_new  # DataFrame contenant de nouvelles données pour la prédiction
Credit_clf_final = st.session_state.Credit_clf_final  # Modèle final de classification
explainer = st.session_state.explainer  # Explicateur du modèle pour l'interprétation des résultats

# --- Création de deux onglets ---
tab1, tab2 = st.tabs(["ID client", "Information manuelle"])

# --- Onglet 1 : Prédiction pour un client avec un ID ---
with tab1:
    st.header("Prédiction pour un client avec ID")
    
    # Explication générale de la section
    st.markdown("""
    **Instructions :**  
    Utilisez cet onglet pour entrer l'ID d'un client spécifique. Le modèle prédira la probabilité de défaut de paiement 
    et affichera les graphiques explicatifs pour mieux comprendre le profil du client.
    """)

    index_client = st.number_input(
        "Entrez l'ID du client (ex : 1, 2, 3, 5)",
        format="%d",
        value=1
    )

    run_btn = st.button(
        'Prédire',
        on_click=None,
        type="primary",
        key='predict_btn1'
    )
    
    if run_btn or "updated_client" in st.session_state:
        if index_client in set(df_new.index):
            if run_btn:
                # Stocker les données originales du client
                data_client = df_new.loc[index_client].fillna(0).to_dict()
                st.session_state['updated_client'] = data_client
            else:
                data_client = st.session_state['updated_client']
            
            # Préparation des données pour SHAP et autres analyses
            X_client = pd.DataFrame([data_client]).fillna(0)

            # Affichage de la jauge avant les graphiques SHAP
            pred_prob = Credit_clf_final.predict_proba(X_client)[0][1]
            plot_gauge(pred_prob)

            # Commentaire pour la jauge
            st.markdown("""
            **Interprétation de la jauge :**  
            La jauge ci-dessus indique la probabilité de défaut de paiement du client. Une valeur proche de 1 signifie un risque élevé, tandis qu'une valeur proche de 0 signifie un faible risque.
            """)

            # --- Volet rétractable pour voir et modifier les caractéristiques du client ---
            with st.expander("Cliquez pour afficher et modifier les caractéristiques du client"):
                st.markdown("""
                **Conseil :**  
                Vous pouvez ajuster manuellement les caractéristiques du client pour observer comment ces modifications influencent la prédiction.
                """)
                
                for feature, value in data_client.items():
                    if isinstance(value, (int, float)):
                        data_client[feature] = st.number_input(
                            label=feature,
                            value=float(value),
                            key=f"input_{feature}"
                        )
                    else:
                        data_client[feature] = st.text_input(
                            label=feature,
                            value=str(value),
                            key=f"input_{feature}"
                        )

                submit_changes = st.button("Mettre à jour les caractéristiques et prédire à nouveau")
            
            if submit_changes:
                # Mettre à jour les nouvelles données dans session_state
                st.session_state['updated_client'] = data_client

                # Relancer la prédiction avec les nouvelles valeurs
                updated_client = pd.DataFrame([data_client])
                pred_prob_updated = Credit_clf_final.predict_proba(updated_client)[0][1]
                plot_gauge(pred_prob_updated)

                # Commentaire pour la jauge après modification
                st.markdown("""
                **Analyse des modifications :**  
                Après avoir modifié les caractéristiques du client, la jauge reflète maintenant la nouvelle probabilité de défaut de paiement. Vous pouvez observer si le risque a augmenté ou diminué.
                """)

                # --- Volet rétractable pour les graphiques SHAP ---
                with st.expander("Voir l'explication de la prédiction pour le client"):
                    shap_plot(explainer, updated_client, 0)
                    st.markdown("""
                    **Interprétation des graphiques SHAP :**  
                    Ces graphiques SHAP montrent les variables les plus influentes sur la décision du modèle. Pour le premier graphique les barres positives, en rouge, augmentent la probabilité de défaut de paiement, et les barres négatives, en bleu, la réduisent. Le second graphique quand à lui donne une lecture globale, où nous voyons la contribution des features que ce soit pour réduire ou augmenter le score-crédit obtenu.
                    """)

                # Autres visualisations
                plot_client(
                    updated_client,
                    explainer,
                    df_reference=df_train,
                    index_client=0  # Utilisation d'un index fictif pour un client modifié
                )
                nan_values(updated_client, index_client=0)
            else:
                # --- Volet rétractable pour les graphiques SHAP avec les valeurs originales ---
                with st.expander("Voir l'explication de la prédiction pour le client"):
                    shap_plot(explainer, df_new, index_client)
                    st.markdown("""
                    **Analyse des résultats originaux :**  
                    Les graphiques ci-dessous montrent l'impact des caractéristiques actuelles du client sur la probabilité de défaut de paiement. Utilisez ces informations pour mieux comprendre le profil du client.
                    """)

                # Autres visualisations
                plot_client(
                    df_new,
                    explainer,
                    df_reference=df_train,
                    index_client=index_client
                )
                st.markdown("""
            **Analyse du profil du client :**  
            Le graphique ci-dessus compare les caractéristiques du client à celles d'autres clients dans la base de données. Cela permet de comprendre où se situe le client par rapport à la population.
            """)
                nan_values(df_new, index_client=index_client)
        else:
            st.write("Client non trouvé dans la base de données")


# --- Onglet 2 : Prédiction pour un nouveau client sans ID ---
with tab2:  # Utilisation du second onglet créé
    st.header("Prédiction pour un nouveau client")

    # Explication de la section
    st.markdown("""
    **Instructions :**  
    Dans cet onglet, vous pouvez entrer les caractéristiques d'un nouveau client manuellement, via un texte ou un fichier CSV. 
    Le modèle prédira la probabilité de défaut de paiement et affichera les graphiques explicatifs correspondants.
    """)

    option = st.selectbox(
        'Comment souhaitez-vous entrer les données ?',
        ('Manuel', 'Texte', 'Fichier CSV')
    )

    data_client = None

    # Cas d'entrée manuel
    if option == 'Manuel':
        with st.expander("Cliquez pour entrer les données manuellement"):
            data_client = {}
            for feature in df_new.columns:  # df_new : dataframe avec les nouvelles données du client
                if df_train[feature].dtype == np.int64:
                    min_values = df_train[feature].min().astype('int')
                    max_values = df_train[feature].max().astype('int')
                    data_client[feature] = st.number_input(
                        str(feature), min_value=min_values, max_value=max_values, step=1
                    )
                else:
                    min_values = df_train[feature].min().astype('float')
                    max_values = df_train[feature].max().astype('float')
                    data_client[feature] = st.number_input(
                        str(feature), min_value=min_values, max_value=max_values, step=0.1
                    )

    # Cas d'entrée texte
    elif option == 'Texte':
        with st.expander("Cliquez pour entrer les données sous forme de texte"):
            texte_client = st.text_area('Entrez les données sous forme de dictionnaire', '''{"Taux_Paiement": 0.03, ... }''')
            if texte_client:
                try:
                    data_client = json.loads(texte_client)  # Convertir le texte en dictionnaire
                except json.JSONDecodeError:
                    st.error("Le format du texte n'est pas valide. Veuillez entrer un dictionnaire valide.")

    # Cas d'entrée via un fichier CSV
    elif option == 'Fichier CSV':
        loader = st.file_uploader("Chargez le fichier CSV")
        if loader is not None:
            try:
                # Charger le fichier CSV
                data_client = pd.read_csv(loader, sep=";", index_col="SK_ID_CURR")

                # Forcer la conversion des colonnes numériques
                for column in data_client.columns:
                    data_client[column] = pd.to_numeric(data_client[column], errors='coerce')

                # Afficher un avertissement si des colonnes sont encore non-numériques
                non_numeric_columns = data_client.select_dtypes(include=['object']).columns
                if len(non_numeric_columns) > 0:
                    st.warning(f"Colonnes contenant des valeurs non-numériques : {list(non_numeric_columns)}")

                st.write("Données après conversion :", data_client)

            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier CSV : {e}")

    # Si les données client ont été correctement récupérées, proposer la prédiction
    if data_client is not None:
        run_btn2 = st.button(
            'Prédire',
            on_click=None,
            type="primary",
            key='predict_btn2'
        )

        if run_btn2:
            if isinstance(data_client, dict):  # Si les données sont au format dict (manuel ou texte)
                data_client_df = pd.DataFrame([data_client])  # Convertir en dataframe
            elif isinstance(data_client, pd.DataFrame):  # Si c'est déjà un dataframe (fichier CSV)
                data_client_df = data_client
            else:
                st.error("Format des données incorrect.")
                data_client_df = None

            if data_client_df is not None:
                # Conversion explicite des types de données pour éviter les erreurs de sérialisation
                data_client_df_cleaned = data_client_df.astype(object).where(pd.notnull(data_client_df), None)

                try:
                    # Convertir le DataFrame en une liste de dictionnaires
                    json_data = json.dumps(data_client_df_cleaned.to_dict(orient='records'))

                    # Appeler la fonction de prédiction et autres visualisations avec les données en JSON
                    execute_API(json_data)

                    # --- Volet rétractable pour les graphiques SHAP pour un nouveau client ---
                    with st.expander("Voir les graphiques SHAP"):
                        shap_plot(explainer, df_new, 0)
                        st.markdown("""
                        **Interprétation des résultats pour le nouveau client :**  
                        Les graphiques ci-dessous vous permettent de voir les principales caractéristiques qui influencent la prédiction de défaut de paiement pour ce nouveau client.
                        """)

                    # Autres visualisations et fonctionnalités
                    plot_client(
                        data_client_df_cleaned,
                        explainer,
                        df_reference=df_train,
                        index_client=0  # Utilisation d'un index fictif (0) pour un nouveau client
                    )

                    # Gestion des valeurs manquantes (nan)
                    nan_values(data_client_df_cleaned, index_client=0)

                except Exception as e:
                    st.error(f"Erreur lors de la conversion des données en JSON : {e}")
