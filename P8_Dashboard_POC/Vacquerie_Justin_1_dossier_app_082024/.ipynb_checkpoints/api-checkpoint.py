from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import joblib

# Charger le modèle sauvegardé
with open('app/model/best_model.pkl', 'rb') as file:
    model = joblib.load(file)

# Initialisation de FastAPI
app = FastAPI()

# Classe de données client avec la nouvelle liste de features
class ClientData(BaseModel):
    Taux_Paiement: float
    Score_Externe_2: float
    Score_Externe_3: float
    Jours_Depuis_Naissance: int
    Montant_Annuite: float
    Somme_Paiements: float
    Moyenne_Paiements_Precedents: float
    Montant_Credit: float
    Moyenne_Jours_Retard_Paiements: float
    Moyenne_Paiements_Approuves: float
    Prix_Biens: float
    Jours_Depuis_Publication_ID: int
    Jours_Emploi: float
    Max_Jours_Credit_Bureau: float
    Taille_Solde_Mensuel_POS: float
    Max_Jours_Entree_Paiement: float
    Jours_Dernier_Changement_Telephone: float
    Montant_Min_Paiement: float
    Max_Jours_Fin_Credit_Bureau: float
    Moyenne_Jours_Entree_Paiement: float
    Moyenne_Credit_Dette_Bureau: float
    Moyenne_Diff_Paiement: float
    Moyenne_Annuite_Approuvee: float
    Moyenne_Contrats_Refuses: float
    Pourcentage_Jours_Emploi: float
    Moyenne_Combinaison_Produits_CashXSelllow: float
    Max_Paiement: float
    Somme_Total_Credit_Bureau: float
    Pourcentage_Annuite_Revenu: float
    Max_Versement_Initial: float
    Pourcentage_Revenu_Credit: float
    Somme_Jours_Entree_Paiement: float
    Min_Credit_Applications_Precedentes: float
    Contrat_Type_Pret_Renouvelable: bool = False
    Evaluation_Regionale_Client_Ville: int = 0
    Type_Occupation_Chauffeurs: bool = False
    Niveau_Education_Superieur: bool = False
    Moyenne_Credit_Bureau_Hypotheque: float = 0.0
    Somme_Paiements_Approuves: float = 0.0
    Jours_Depuis_Enregistrement: float = 0.0
    Moyenne_DPD_Defaut_POS: float = 0.0
    Somme_Solde_Mensuel_Bureau_Credit: float = 0.0
    Moyenne_Rendement_Precedent_Eleve: float = 0.0
    Moyenne_Rendement_Precedent_Faible: float = 0.0
    Pourcentage_Max_Credit_Approuve: float = 0.0
    Population_Relative_Region: float = 0.0
    Moyenne_Pourcentage_Credit_Applications_Precedentes: float = 0.0
    Nombre_Versions_Echeancier_Installation: float = 0.0
    Somme_Limites_Credit_Bureau: float = 0.0
    Max_Credit_Bureau: float = 0.0
    Max_Jours_Retard_Paiements_Installation: float = 0.0
    Statut_Familial_Marie: bool = False
    Moyenne_Type_Paiement_Precedent_XNA: float = 0.0
    Moyenne_Jours_Credit_Bureau: float = 0.0
    Indicateur_Possession_Voiture: int = 0
    Moyenne_Type_Credit_Bureau_Microcredit: float = 0.0
    Max_Jours_Decision_Approuvee: float = 0.0
    Somme_Dettes_Credit_Bureau: float = 0.0
    Moyenne_Pourcentage_Paiement_Installation: float = 0.0
    Moyenne_Type_Client_Precedent_Nouveau: float = 0.0
    Moyenne_Montant_Paiement: float = 0.0
    Moyenne_Credit_Retard_Bureau: float = 0.0
    Moyenne_Jours_Retard_Paiements_Installation: float = 0.0
    Moyenne_Credit_Bureau: float = 0.0
    Moyenne_Credits_Actifs_Bureau: float = 0.0
    Nombre_Demandes_Credit_Bureau_Trimestre: float = 0.0
    Somme_Paiements_Precedents: float = 0.0
    Moyenne_Solde_Mensuel_POS: float = 0.0
    Moyenne_Jours_Fin_Credit_Bureau: float = 0.0
    Revenu_Par_Personne: float = 0.0

def make_prediction(input_data, threshold=0.4):
    try:
        input_data = np.array(input_data).reshape(1, -1)
        probability = model.predict_proba(input_data)
        
        # Extraire la probabilité pour la classe positive (classe 1)
        probability_class_1 = probability[0][1]
        
        # Utiliser le seuil pour déterminer la classe prédite
        prediction = (probability_class_1 >= threshold).astype(int)
        return prediction, probability_class_1
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")


# Message d'accueil
@app.get("/")
def read_root():
    return {"message": "Bonjour, vous êtes bien sur l'application de scoring, hébergée sur Heroku. "
                       "Cette API permet de prédire la probabilité de défaut de paiement pour un client "
                       "en fonction de ses caractéristiques. Envoyez une requête POST à /predict pour obtenir une prédiction."}

@app.post("/predict")
def predict(client_data: ClientData):
    try:
        input_data = [
            client_data.Taux_Paiement,
            client_data.Score_Externe_2,
            client_data.Score_Externe_3,
            client_data.Jours_Depuis_Naissance,
            client_data.Montant_Annuite,
            client_data.Somme_Paiements,
            client_data.Moyenne_Paiements_Precedents,
            client_data.Montant_Credit,
            client_data.Moyenne_Jours_Retard_Paiements,
            client_data.Moyenne_Paiements_Approuves,
            client_data.Prix_Biens,
            client_data.Jours_Depuis_Publication_ID,
            client_data.Jours_Emploi,
            client_data.Max_Jours_Credit_Bureau,
            client_data.Taille_Solde_Mensuel_POS,
            client_data.Max_Jours_Entree_Paiement,
            client_data.Jours_Dernier_Changement_Telephone,
            client_data.Montant_Min_Paiement,
            client_data.Max_Jours_Fin_Credit_Bureau,
            client_data.Moyenne_Jours_Entree_Paiement,
            client_data.Moyenne_Credit_Dette_Bureau,
            client_data.Moyenne_Diff_Paiement,
            client_data.Moyenne_Annuite_Approuvee,
            client_data.Moyenne_Contrats_Refuses,
            client_data.Pourcentage_Jours_Emploi,
            client_data.Moyenne_Combinaison_Produits_CashXSelllow,
            client_data.Max_Paiement,
            client_data.Somme_Total_Credit_Bureau,
            client_data.Pourcentage_Annuite_Revenu,
            client_data.Max_Versement_Initial,
            client_data.Pourcentage_Revenu_Credit,
            client_data.Somme_Jours_Entree_Paiement,
            client_data.Min_Credit_Applications_Precedentes,
            client_data.Contrat_Type_Pret_Renouvelable,
            client_data.Evaluation_Regionale_Client_Ville,
            client_data.Type_Occupation_Chauffeurs,
            client_data.Niveau_Education_Superieur,
            client_data.Moyenne_Credit_Bureau_Hypotheque,
            client_data.Somme_Paiements_Approuves,
            client_data.Jours_Depuis_Enregistrement,
            client_data.Moyenne_DPD_Defaut_POS,
            client_data.Somme_Solde_Mensuel_Bureau_Credit,
            client_data.Moyenne_Rendement_Precedent_Eleve,
            client_data.Moyenne_Rendement_Precedent_Faible,
            client_data.Pourcentage_Max_Credit_Approuve,
            client_data.Population_Relative_Region,
            client_data.Moyenne_Pourcentage_Credit_Applications_Precedentes,
            client_data.Nombre_Versions_Echeancier_Installation,
            client_data.Somme_Limites_Credit_Bureau,
            client_data.Max_Credit_Bureau,
            client_data.Max_Jours_Retard_Paiements_Installation,
            client_data.Statut_Familial_Marie,
            client_data.Moyenne_Type_Paiement_Precedent_XNA,
            client_data.Moyenne_Jours_Credit_Bureau,
            client_data.Indicateur_Possession_Voiture,
            client_data.Moyenne_Type_Credit_Bureau_Microcredit,
            client_data.Max_Jours_Decision_Approuvee,
            client_data.Somme_Dettes_Credit_Bureau,
            client_data.Moyenne_Pourcentage_Paiement_Installation,
            client_data.Moyenne_Type_Client_Precedent_Nouveau,
            client_data.Moyenne_Montant_Paiement,
            client_data.Moyenne_Credit_Retard_Bureau,
            client_data.Moyenne_Jours_Retard_Paiements_Installation,
            client_data.Moyenne_Credit_Bureau,
            client_data.Moyenne_Credits_Actifs_Bureau,
            client_data.Nombre_Demandes_Credit_Bureau_Trimestre,
            client_data.Somme_Paiements_Precedents,
            client_data.Moyenne_Solde_Mensuel_POS,
            client_data.Moyenne_Jours_Fin_Credit_Bureau,
            client_data.Revenu_Par_Personne
        ]

        # Faire la prédiction et obtenir les probabilités
        prediction, probability = make_prediction(input_data, threshold=0.4)

        return {"prediction": int(prediction), "probability": float(probability)}
    
    except HTTPException as e:
        raise e
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")
