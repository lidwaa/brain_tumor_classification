import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

model = load_model('../static/advanced_tumor_classification_model.keras')

class_names = ['glioma', 'meningioma', 'pituitary', 'no tumor']

def prepare_image(image_path):
    """
    Charge et prépare l'image pour la prédiction : redimensionnement, normalisation, ajout des dimensions nécessaires.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=-1)
    img_input = np.expand_dims(img_expanded, axis=0)
    return img_input

def predict_image(model, image_path):
    """
    Prédit la classe d'une image donnée à l'aide du modèle.
    """
    img_input = prepare_image(image_path)
    
    prediction = model.predict(img_input)
    
    predicted_class_index = np.argmax(prediction, axis=1)
    
    return class_names[predicted_class_index[0]]

if __name__ == '__main__':
    image_path = 'C:/Users/WALID/Downloads/notumor.jpeg'  

    if os.path.exists(image_path):
        predicted_class_name = predict_image(model, image_path)
        print(f"Classe prédite pour l'image {image_path}: {predicted_class_name}")
    else:
        print(f"L'image spécifiée {image_path} n'existe pas.")
