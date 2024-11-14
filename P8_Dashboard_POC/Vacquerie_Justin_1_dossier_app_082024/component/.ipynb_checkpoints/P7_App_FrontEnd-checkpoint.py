import streamlit as st
import requests
import json
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
from math import pi
from sklearn.preprocessing import MinMaxScaler

# Fonction sans API pour afficher la prédiction et les probabilités de défaut de paiement pour un client spécifique
def execute_noAPI(df, index_client, model):
    """ 
    Fonction générant les colonnes dans l'interface Streamlit montrant la prédiction du défaut de paiement.
    """
    st.subheader('Difficultés du client : ')
    predict_proba = model.predict_proba([df.drop(columns='TARGET').fillna(0).loc[index_client]])[:, 1]
    predict_target = (predict_proba >= 0.4).astype(int)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Difficultés", str(np.where(df['TARGET'].loc[index_client] == 0, 'NON', 'OUI')))
    col2.metric("Difficultés Prédites", str(np.where(predict_target == 0, 'NON', 'OUI'))[2:-2])
    col3.metric("Probabilité", predict_proba.round(2))

# Interface Streamlit pour saisir l'ID client
st.title("Prédiction du défaut de paiement")

# Saisie de l'ID client
sk_id_curr = st.text_input("Entrez l'ID du client pour obtenir la prédiction :")

# Bouton pour lancer la prédiction
if st.button("Obtenir la prédiction via l'API"):
    if sk_id_curr:
        try:
            # Vérification si les données sont déjà chargées
            if 'df_new' in st.session_state:
                df_new = st.session_state.df_new
            else:
                st.error("Les données client ne sont pas chargées.")
                st.stop()

            # Récupérer les données du client correspondant à l'ID
            client_data = df_new.loc[int(sk_id_curr)].fillna(0).to_dict()

            # URL de l'API
            api_url = "https://app-scoring-p7-b4207842daea.herokuapp.com/predict"

            # Envoi de la requête POST à l'API avec toutes les caractéristiques du client
            response = requests.post(api_url, json=client_data)

            # Vérification du statut de la réponse
            if response.status_code == 200:
                result = response.json()
                st.write(f"Prédiction : {'Oui' if result['prediction'] == 1 else 'Non'}")
                st.write(f"Probabilité de défaut : {result['probability'] * 100:.2f}%")
            else:
                st.error(f"Erreur : {response.json()['detail']}")
        
        except Exception as e:
            st.error(f"Erreur lors de la requête à l'API : {e}")
    else:
        st.error("Veuillez entrer un ID client valide.")

def execute_API(df):
    st.subheader('Client difficulties : ')
    
    # Effectuer la requête
    request = requests.post(
        url="https://app-scoring-p7-b4207842daea.herokuapp.com/predict",
        data=json.dumps(df),
        headers={"Content-Type": "application/json"}
    )
    
    # Vérifier si la requête a réussi
    if request.status_code == 200:
        response_json = request.json()  # Obtenez la réponse JSON
        
        # Afficher la réponse JSON complète pour diagnostic
        st.write(response_json)
        
        # s'assurer que les clés sont présentes
        if "prediction" in response_json and "probability" in response_json:
            prediction = response_json["prediction"]
            probability = round(response_json["probability"], 2)
            
            # Afficher les résultats
            col1, col2 = st.columns(2)
            col1.metric("Predicted Difficulties", str(np.where(prediction == 0, 'NO', 'YES')))
            col2.metric("Probability of default", probability)
        else:
            st.error("Les clés 'prediction' ou 'probability' sont manquantes dans la réponse.")
    else:
        st.error(f"Erreur avec la requête API : {request.status_code}")


def shap_plot(explainer, df, index_client=0):
    """
    Cette fonction génère un graphique waterfall des valeurs SHAP pour un client spécifique.
    """

    # Vérification que l'index du client existe dans le DataFrame
    if index_client not in df.index:
        st.error(f"L'index client {index_client} n'existe pas dans le DataFrame.")
        return

    # Sélection des données pour le client
    X = df.fillna(0).loc[[index_client]]

    try:
        # Appel de l'explainer SHAP pour obtenir un objet Explanation
        shap_values = explainer(X)

        # Génération du waterfall plot pour visualiser les valeurs SHAP d'un client spécifique
        st.write("Valeurs SHAP pour ce client :")
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap_values[0], show=False)  # Waterfall plot
        
        # Affichage du graphique dans Streamlit
        st.pyplot(fig)
        plt.clf()

        # --- Ajout du bar plot pour l'importance globale des features ---
        st.write("Importance globale des caractéristiques :")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False)  # Bar plot pour l'importance globale des features
        st.pyplot(fig)
        plt.clf()

    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'appel à l'explainer SHAP : {str(e)}")
        st.error("Vérifiez que les données passées à l'explainer sont correctes.")


def plot_client(df, explainer, df_reference, index_client=0):
    """ 
    Génère des visualisations améliorées pour comprendre la prédiction de défaut de prêt pour un client spécifique.
    """
    
    try:
        shap_values = explainer(df.fillna(0).loc[[index_client]])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Classe 1 = défaut de paiement
        shap_values = shap_values.values.flatten()
        shap_importance = pd.Series(shap_values, index=df.columns).abs().sort_values(ascending=False)

    except Exception as e:
        st.error(f"Erreur lors du calcul des valeurs SHAP : {e}")
        return

    st.subheader('Explication : Top 6 caractéristiques discriminantes')
    col1, col2 = st.columns(2)

    with col1:
        for feature in list(shap_importance.index[:6])[:3]:
            try:
                plt.figure(figsize=(5, 5))
                sns.set_style("whitegrid")

                if df_reference[feature].nunique() == 2:
                    figInd = sns.barplot(df_reference[['TARGET', feature]].fillna(0).groupby(
                        'TARGET').value_counts(normalize=True).reset_index(), x=feature, y=0, hue='TARGET', 
                        palette="coolwarm", alpha=0.8)
                    plt.ylabel('Fréquence des clients')

                    plt.scatter(y=df[feature].loc[index_client] + 0.1, x=feature, marker='o', s=150, color="r", edgecolor="black", zorder=5)
                    figInd.annotate('Client ID:\n{}'.format(index_client), xy=(feature, df[feature].loc[index_client] + 0.1),
                                    xytext=(0, 40), textcoords='offset points', ha='center', va='bottom',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black"), 
                                    arrowprops=dict(arrowstyle="->", color="black"))

                    legend_handles, _ = figInd.get_legend_handles_labels()
                    figInd.legend(legend_handles, ['Non', 'Oui'], title="Défaut de prêt", loc="upper right", frameon=False)
                    st.pyplot(figInd.figure)
                    plt.close()

                else:
                    figInd = sns.boxplot(data=df_reference, y=feature, x='TARGET', showfliers=False, width=0.3, palette="coolwarm")
                    plt.xlabel('Défaut de prêt')
                    figInd.set_xticklabels(["Non", "Oui"])

                    plt.scatter(y=df[feature].loc[index_client], x=0.5, marker='o', s=150, color="r", edgecolor="black", zorder=5)
                    figInd.annotate('Client ID:\n{}'.format(index_client), xy=(0.5, df[feature].loc[index_client]),
                                    xytext=(0, 40), textcoords='offset points', ha='center', va='bottom',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black"), 
                                    arrowprops=dict(arrowstyle="->", color="black"))

                    figInd.axhline(y=df_reference[df_reference['TARGET'] == 0][feature].mean(), linestyle='--', color="#1f77b4")
                    figInd.axhline(y=df_reference[df_reference['TARGET'] == 1][feature].mean(), linestyle='--', color="#ff7f0e")

                    colors = ["#1f77b4", "#ff7f0e"]
                    lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='--') for c in colors]
                    labels = ["Moyenne Sans Défaut", "Moyenne Avec Défaut"]
                    plt.legend(lines, labels, title="Moyennes des clients", loc="upper right", frameon=False)
                    st.pyplot(figInd.figure)
                    plt.close()

            except Exception as e:
                st.error(f"Erreur lors de la génération du graphique pour la caractéristique {feature} : {e}")

    with col2:
        for feature in list(shap_importance.index[:6])[3:]:
            try:
                plt.figure(figsize=(5, 5))
                sns.set_style("whitegrid")

                if df_reference[feature].nunique() == 2:
                    figInd = sns.barplot(df_reference[['TARGET', feature]].fillna(0).groupby(
                        'TARGET').value_counts(normalize=True).reset_index(), x=feature, y=0, hue='TARGET', 
                        palette="coolwarm", alpha=0.8)
                    plt.ylabel('Fréquence des clients')

                    plt.scatter(y=df[feature].loc[index_client] + 0.1, x=feature, marker='o', s=150, color="r", edgecolor="black", zorder=5)
                    figInd.annotate('Client ID:\n{}'.format(index_client), xy=(feature, df[feature].loc[index_client] + 0.1),
                                    xytext=(0, 40), textcoords='offset points', ha='center', va='bottom',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black"), 
                                    arrowprops=dict(arrowstyle="->", color="black"))

                    legend_handles, _ = figInd.get_legend_handles_labels()
                    figInd.legend(legend_handles, ['Non', 'Oui'], title="Défaut de prêt", loc="upper right", frameon=False)
                    st.pyplot(figInd.figure)
                    plt.close()

                else:
                    figInd = sns.boxplot(data=df_reference, y=feature, x='TARGET', showfliers=False, width=0.3, palette="coolwarm")
                    plt.xlabel('Défaut de prêt')
                    figInd.set_xticklabels(["Non", "Oui"])

                    plt.scatter(y=df[feature].loc[index_client], x=0.5, marker='o', s=150, color="r", edgecolor="black", zorder=5)
                    figInd.annotate('Client ID:\n{}'.format(index_client), xy=(0.5, df[feature].loc[index_client]),
                                    xytext=(0, 40), textcoords='offset points', ha='center', va='bottom',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black"), 
                                    arrowprops=dict(arrowstyle="->", color="black"))

                    figInd.axhline(y=df_reference[df_reference['TARGET'] == 0][feature].mean(), linestyle='--', color="#1f77b4")
                    figInd.axhline(y=df_reference[df_reference['TARGET'] == 1][feature].mean(), linestyle='--', color="#ff7f0e")

                    colors = ["#1f77b4", "#ff7f0e"]
                    lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='--') for c in colors]
                    labels = ["Moyenne Sans Défaut", "Moyenne Avec Défaut"]
                    plt.legend(lines, labels, title="Moyennes des clients", loc="upper right", frameon=False)
                    st.pyplot(figInd.figure)
                    plt.close()

            except Exception as e:
                st.error(f"Erreur lors de la génération du graphique pour la caractéristique {feature} : {e}")
                

    # --- Analysis des valeurs manquantes ---

def nan_values(df, index_client=0):
    # Utiliser pd.isna() 
    if df.loc[index_client].isna().any():
        st.subheader('Attention : colonnes avec des valeurs manquantes')
        nan_col = []
        for feature in df.columns:
            # Vérifier les valeurs manquantes pour chaque caractéristique
            if pd.isna(df.loc[index_client, feature]):
                nan_col.append(feature)

        col1, col2 = st.columns(2)
        with col1:
            st.table(data=pd.DataFrame(nan_col, columns=['FEATURES WITH MISSING VALUES']))
        with col2:
            st.write('Toutes les valeurs manquantes ont été remplacées par 0.')
    else:
        st.subheader('Il n\'y a pas de valeurs manquantes dans la base de données concernant ce client')


      # --- Création d'une jauge pour simplifier la lecture des résultats ---
def plot_gauge(pred_prob, threshold=0.4, title="Prédiction de la probabilité du client", xlabel="Probabilité"):
    fig, ax = plt.subplots(figsize=(8, 4))

    # Définir les limites de la jauge
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    
    # Définir un gradient de couleurs
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    cmap = plt.get_cmap('RdYlGn_r')

    # Afficher le gradient
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, -0.5, 0.5])

    # Position de l'aiguille
    ax.plot([pred_prob, pred_prob], [-0.5, 0.5], color='black', lw=6)

    # Ajouter la ligne pointillée pour le seuil
    ax.plot([threshold, threshold], [-0.5, 0.5], color='blue', lw=2, linestyle='--')

    # Ajouter des annotations pour indiquer la probabilité prédite
    ax.text(pred_prob, 0.55, f'{pred_prob:.2f}', horizontalalignment='center', fontsize=16, color='black', fontweight='bold')

    # Ajuster la position du seuil légèrement plus bas
    plt.figtext(0.5, 0.03, f'Seuil ({threshold})', ha='center', fontsize=14, color='blue', fontweight='bold')

    # Ajouter des graduations de probabilité
    for i in np.linspace(0, 1, 11):
        ax.text(i, -0.6, f'{i:.1f}', horizontalalignment='center', fontsize=12, color='gray')

    # Retirer les axes inutiles
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Ajuster la position du titre
    plt.title(title, fontsize=16, fontweight='bold', pad=40)

    plt.box(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    st.pyplot(fig)


#---------- Création d'une fonction scatterplot interactive--------

# Fonction pour afficher un scatter plot interactif
def scatter_plot_interactif(df):
#    st.title("Analyse bivariée : Nuage de points")

    # Sélection des colonnes pour les axes
    colonnes = df.columns.tolist()
    x_colonne = st.selectbox("Sélectionnez la colonne pour l'axe X", colonnes)
    y_colonne = st.selectbox("Sélectionnez la colonne pour l'axe Y", colonnes)
    target_colonne = st.selectbox("Sélectionnez la colonne pour la couleur des points (target)", colonnes)

    # Créer le scatter plot si les colonnes sont sélectionnées
    if x_colonne and y_colonne and target_colonne:
        # Choisir un style esthétique avec Seaborn
        sns.set(style="whitegrid")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Création du scatter plot avec personnalisation
        sc = ax.scatter(df[x_colonne], df[y_colonne], 
                        c=df[target_colonne], cmap='viridis', 
                        s=100, edgecolor='white', alpha=0.7)

        ax.set_xlabel(x_colonne, fontsize=12)
        ax.set_ylabel(y_colonne, fontsize=12)
        ax.set_title(f"Scatter Plot : {x_colonne} vs {y_colonne} (Colorié par {target_colonne})", fontsize=16, weight='bold')

        # Ajout d'une grille pour une meilleure lisibilité
        ax.grid(True, linestyle='--', alpha=0.7)

        # Ajout de la barre de couleur pour représenter les valeurs de la target
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(target_colonne, rotation=270, labelpad=15)

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)


# --------- fonction pour afficher la distribution de la feature selon son type

def univariate_analysis(df):
#    st.title("Analyse univariée : Distribution des données")

    # Sélection d'une colonne pour l'analyse
    colonnes = df.columns.tolist()
    colonne = st.selectbox("Sélectionnez la colonne à analyser", colonnes)

    # Vérification du type de colonne (numérique ou catégorielle)
    if colonne:
        if df[colonne].dtype == 'object':
            # Analyse pour des variables catégorielles (ex : un graphique à barres)
            distribution = df[colonne].value_counts()
            fig, ax = plt.subplots()
            bars = ax.bar(distribution.index, distribution.values)
            ax.set_xlabel(colonne)
            ax.set_ylabel("Fréquence")
            ax.set_title(f"Distribution des catégories pour {colonne}")
            plt.xticks(rotation=90)  # Pour éviter que les étiquettes ne se chevauchent

            # Ajout des étiquettes de valeurs au-dessus des barres
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), 
                        ha='center', va='bottom')

            st.pyplot(fig)

        else:
            # Analyse pour des variables numériques (ex : histogramme)
            fig, ax = plt.subplots()
            n, bins, patches = ax.hist(df[colonne], bins=30)
            ax.set_xlabel(colonne)
            ax.set_ylabel("Fréquence")
            ax.set_title(f"Distribution des valeurs pour {colonne}")

            # Ajout des étiquettes de valeurs au-dessus des barres de l'histogramme
            for i in range(len(patches)):
                ax.text(patches[i].get_x() + patches[i].get_width() / 2, n[i], 
                        int(n[i]), ha='center', va='bottom')

            st.pyplot(fig)



# Fonction pour créer un radar plot avec normalisation
def radar_plot(df, client_id, axes_options):
    """
    Fonction pour créer un radar plot comparant un client avec la moyenne de tous les clients,
    en appliquant une normalisation min-max pour éviter les problèmes d'échelle.
    
    Paramètres:
    df : DataFrame
        Le dataframe contenant les données des clients, incluant une colonne 'SK_ID_CURR' et les colonnes des axes à étudier.
    client_id : int
        L'ID du client à comparer.
    axes_options : list
        La liste des colonnes correspondant aux axes disponibles pour le radar plot.
    """
    
    # Limiter la sélection par défaut aux axes disponibles
    default_axes = [axis for axis in axes_options if axis in df.columns][:8]
    
    # Sélection des axes d'intérêt
    selected_axes = st.multiselect("Choisissez les axes à étudier", axes_options, default=default_axes)
    
    if len(selected_axes) < 3:
        st.warning("Veuillez sélectionner au moins 3 axes pour le radar plot.")
        return

    # Filtrer les données du client
    client_data = df[df.index == client_id]
    
    if client_data.empty:
        st.warning(f"Aucun client trouvé avec l'ID {client_id}.")
        return
    
    # Normalisation des données avec Min-Max Scaler
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df[selected_axes]), columns=selected_axes)
    client_data_normalized = pd.DataFrame(scaler.transform(client_data[selected_axes]), columns=selected_axes)
    
    # Moyenne des clients normalisée
    mean_values = df_normalized.mean()

    # Valeurs du client normalisées
    client_values = client_data_normalized.values.flatten()

    # Préparation des données pour le radar plot
    categories = selected_axes
    num_vars = len(categories)

    # Créer des angles pour chaque axe
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # Créer le radar plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Afficher la moyenne de tous les clients
    mean_values = mean_values.tolist()
    mean_values += mean_values[:1]
    ax.fill(angles, mean_values, color='blue', alpha=0.1)
    ax.plot(angles, mean_values, color='blue', linewidth=2, linestyle='solid', label="Moyenne des clients")

    # Afficher les valeurs du client
    client_values = client_values.tolist()
    client_values += client_values[:1]
    ax.fill(angles, client_values, color='red', alpha=0.3)
    ax.plot(angles, client_values, color='red', linewidth=2, linestyle='solid', label=f"Client {client_id}")

    # Ajouter les étiquettes des axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Légende
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Afficher le radar plot dans Streamlit
    st.pyplot(fig)
