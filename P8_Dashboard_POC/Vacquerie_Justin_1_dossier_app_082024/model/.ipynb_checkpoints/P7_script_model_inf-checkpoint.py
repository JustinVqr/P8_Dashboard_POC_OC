import joblib
import numpy as np

# Charger le modèle sauvegardé
with open('app/model/best_model.pkl', 'rb') as file:
    model = joblib.load(file)

def make_prediction(input_data, threshold=0.4):
    """
    Fonction pour faire des prédictions avec le modèle chargé.
    Retourne à la fois la prédiction et la probabilité associée.
    
    :param input_data: array-like, shape (n_samples, n_features)
    :param threshold: seuil de décision pour la classification
    :return: tuple contenant la valeur prédite et la probabilité associée
    """
    input_data = np.array(input_data).reshape(1, -1)
    probability = model.predict_proba(input_data)
    
    # Extraire la probabilité pour la classe positive (classe 1)
    probability_class_1 = probability[0][1]
    
    # Utiliser le seuil pour déterminer la classe prédite
    prediction = (probability_class_1 >= threshold).astype(int)
    
    return prediction, probability_class_1
