import os
import cv2
import numpy as np

# Taille des images
IMG_SIZE = 128
classes = ["glioma", "meningioma", "pituitary", "notumor"]

def load_data(data_dir, classes):
    data = []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                # Charger l'image en niveaux de gris
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # Redimensionner
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                # Ajouter l'image et le label à la liste
                data.append([img_resized, label])
            except Exception as e:
                print(f"Erreur lors du chargement de l'image : {img_path}, {e}")
    return data

def preprocess_data(data_dir):
    # Charger les données
    data = load_data(data_dir, classes)
    
    # Séparer les images et les labels
    X, y = zip(*data)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalisation
    y = np.array(y)
    
    return X, y
