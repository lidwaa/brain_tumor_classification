import tensorflow as tf
from keras import layers, models
from data_preprocessing import preprocess_data
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

data_dir = "data"


X, y = preprocess_data(data_dir)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_advanced_model():
    model = models.Sequential([

        # Bloc Convolutionnel 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloc Convolutionnel 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloc Convolutionnel 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Bloc Convolutionnel 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Couches Fully Connected
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Couche de sortie
        layers.Dense(4, activation='softmax') 
    ])
    return model

model = create_advanced_model()

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

predictions = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=["glioma", "meningioma", "pituitary", "notumor"]))

conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["glioma", "meningioma", "pituitary", "no tumor"], yticklabels=["glioma", "meningioma", "pituitary", "notumor"])
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()

def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))

    # Courbe de perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perte - Entraînement')
    plt.plot(history.history['val_loss'], label='Perte - Validation')
    plt.title("Évolution de la perte")
    plt.xlabel("Époque")
    plt.ylabel("Perte")
    plt.legend()

    # Courbe d'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Précision - Entraînement')
    plt.plot(history.history['val_accuracy'], label='Précision - Validation')
    plt.title("Évolution de la précision")
    plt.xlabel("Époque")
    plt.ylabel("Précision")
    plt.legend()

    plt.show()

plot_learning_curves(history)

model.save("static/advanced_tumor_classification_model.keras")

def predict_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (128, 128))
    image_input = np.array(image_resized).reshape(1, 128, 128, 1) / 255.0  
    prediction = model.predict(image_input)
    predicted_class = ["glioma", "meningioma", "pituitary", "no tumor"][np.argmax(prediction)]  
    print(f"Classe prédite : {predicted_class}")

# Exemple d'utilisation pour prédire une nouvelle image
# predict_image("path/to/new/image.png")
