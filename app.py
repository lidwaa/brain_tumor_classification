from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialisation de l'application Flask
app = Flask(__name__)

# Dossier pour stocker les images téléchargées temporairement
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Charger le modèle
model = load_model("advanced_tumor_classification_model.keras")

# Taille des images pour le modèle
IMG_SIZE = 128
classes = ["glioma", "meningioma", "pituitary", "notumor"]

# Créer une fonction pour préparer l'image pour la prédiction
def prepare_image(image_file):
    # Charger l'image en niveaux de gris
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Redimensionner
    img_resized = img_to_array(img_resized)  # Convertir en tableau numpy
    img_resized = np.expand_dims(img_resized, axis=0)  # Ajouter une dimension pour la batch
    img_resized = img_resized / 255.0  # Normalisation
    return img_resized

# Route principale
@app.route('/')
def index():
    return render_template('index.html')

# Route pour télécharger et classifier l'image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Chemin pour l'image téléchargée
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')

    # Sauvegarder l'image dans le dossier 'uploads' et remplacer la précédente
    file.save(image_path)

    # Préparer l'image et faire la prédiction
    img = prepare_image(image_path)
    predictions = model.predict(img)
    predicted_class = classes[np.argmax(predictions)]
    
    # Retourner le résultat à l'utilisateur
    return render_template('index.html', prediction=predicted_class, image_path=image_path)

if __name__ == '__main__':
    # Créer le dossier uploads si il n'existe pas déjà
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
