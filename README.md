# Classification des Tumeurs Cérébrales

Ce projet est une application web permettant de classifier des images de tumeurs cérébrales en utilisant un modèle d'apprentissage profond. Les types de tumeurs pris en charge sont : gliome, méningiome, hypophysaire et non-tumoral.

## Table des matières

- [Description](#description)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Dataset](#dataset)
- [Contributeurs](#contributeurs)
- [Liens](#liens)

## Description

Cette application utilise un modèle de classification d'images développé avec TensorFlow et Keras. L'interface web est construite avec Flask, et elle permet aux utilisateurs d'uploader une image pour obtenir une prédiction sur le type de tumeur détecté.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les éléments suivants :

- Python 3.8 ou plus récent
- Pip pour gérer les dépendances

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/lidwaa/brain_tumor_classification.git
   cd brain_tumor_classification
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Téléchargez le modèle et placez-le dans le répertoire suivant :
   ```
   static/advanced_tumor_classification_model.keras
   ```

4. Lancez l'application :
   ```bash
   python app.py
   ```

5. Ouvrez un navigateur web et accédez à `http://127.0.0.1:5000`.

## Utilisation

1. Chargez une image au format supporté (par exemple, JPEG, PNG).
2. Cliquez sur "Télécharger et classifier".
3. Visualisez le type de tumeur prédit.

## Structure du projet

```
.
├── app.py                   # Application Flask
├── model/
│   ├── data_preprocessing.py # Prétraitement des données
│   ├── model.py             # Création et entraînement du modèle
│   └── predict.py           # Prédiction à partir d'images
├── requirements.txt         # Liste des dépendances
├── static/
│   ├── style.css            # Style de l'application
│   ├── uploads/             # Images uploadées par les utilisateurs
│   └── advanced_tumor_classification_model.keras # Modèle pré-entraîné
├── templates/
│   └── index.html           # Interface utilisateur
└── README.md                # Documentation du projet
```

## Dataset

Le dataset utilisé pour entraîner le modèle est disponible sur Kaggle :  
[Brain Tumor Classification Dataset Optimized](https://www.kaggle.com/datasets/kamaldehbi/brain-tumor-classification-dataset-optimized/data)

Ce dataset contient des images organisées par type de tumeur (gliome, méningiome, hypophysaire, et non-tumoral), prêtes pour le prétraitement et l'entraînement.

## Contributeurs

- **Kamal Dehbi** - [@kamaLc73](https://github.com/kamaLc73)  
- **Walid Elbachar** - [@lidwaa](https://github.com/lidwaa)

## Liens

- Dépôt GitHub : [https://github.com/lidwaa/brain_tumor_classification.git](https://github.com/lidwaa/brain_tumor_classification.git)  
- Dataset : [Brain Tumor Classification Dataset Optimized](https://www.kaggle.com/datasets/kamaldehbi/brain-tumor-classification-dataset-optimized/data)
