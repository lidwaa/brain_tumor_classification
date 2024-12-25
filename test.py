from tensorflow.keras.models import load_model

# Charger ton modèle depuis un fichier .h5
model = load_model('tumor_classification_model.h5')
# Sauvegarder le modèle au format .keras
model.save('my_model.keras')
