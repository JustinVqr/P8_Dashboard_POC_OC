import streamlit as st
import requests

# URL de l'API FastAPI déployée sur Heroku
API_URL = "https://votre-app.herokuapp.com/predict"

st.title("Prédiction de Défaut de Paiement")
st.write("Entrez l'identifiant du client pour obtenir une prédiction.")

# Entrée de l'identifiant client
client_id = st.number_input("Identifiant Client", min_value=100001, max_value=999999, step=1)

# Lorsqu'un utilisateur soumet l'identifiant, envoyez une requête à l'API
if st.button("Obtenir la Prédiction"):
    response = requests.post(API_URL, json={"SK_ID_CURR": int(client_id)})

    if response.status_code == 200:
        prediction_data = response.json()
        st.write(f"Prédiction : {prediction_data['prediction']}")
        st.write(f"Probabilité : {prediction_data['probability']}")
    else:
        st.write("Erreur : Impossible de récupérer les données pour cet identifiant.")