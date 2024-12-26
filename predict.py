import numpy as np
import cv2
from keras.models import load_model
import os


# Charger le modèle sauvegardé
model = load_model('advanced_tumor_classification_model.keras')

# Liste des classes possibles (tu dois les adapter à ton modèle)
class_names = ['glioma', 'meningioma', 'pituitary', 'no tumor']

def prepare_image(image_path):
    """
    Charge et prépare l'image pour la prédiction : redimensionnement, normalisation, ajout des dimensions nécessaires.
    """
    # Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Redimensionner l'image à la taille d'entrée du modèle (par exemple 128x128)
    img_resized = cv2.resize(img, (128, 128))
    # Normaliser l'image (scaler entre 0 et 1)
    img_normalized = img_resized / 255.0
    # Ajouter la dimension du canal (si l'image est en niveaux de gris)
    img_expanded = np.expand_dims(img_normalized, axis=-1)
    # Ajouter la dimension du batch
    img_input = np.expand_dims(img_expanded, axis=0)
    return img_input

def predict_image(model, image_path):
    """
    Prédit la classe d'une image donnée à l'aide du modèle.
    """
    # Préparer l'image
    img_input = prepare_image(image_path)
    
    # Faire la prédiction
    prediction = model.predict(img_input)
    
    # Obtenir la classe prédite (classe avec la probabilité la plus élevée)
    predicted_class_index = np.argmax(prediction, axis=1)
    
    # Retourner le nom de la classe à partir de l'indice
    return class_names[predicted_class_index[0]]

if __name__ == '__main__':
    # Spécifier le chemin de l'image à tester
    image_path = 'C:/Users/WALID/Downloads/notumor.jpeg'  # Remplace par le chemin de l'image que tu veux tester

    # Vérifier si l'image existe
    if os.path.exists(image_path):
        # Prédire la classe de l'image
        predicted_class_name = predict_image(model, image_path)
        print(f"Classe prédite pour l'image {image_path}: {predicted_class_name}")
    else:
        print(f"L'image spécifiée {image_path} n'existe pas.")
