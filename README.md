# Détection de campagnes de désinformation — projet Machine Learning AI for Good

Ce projet est un exemple pédagogique complet pour GitHub et Google Colab.
Il utilise des **données imaginées** pour entraîner un modèle qui détecte des publications ressemblant à une **campagne coordonnée de désinformation**.

> Important : ce projet est éducatif. Il ne doit pas servir à censurer automatiquement des personnes. Un vrai système devrait inclure vérification humaine, transparence, audit, analyse de biais et protection de la vie privée.

## Objectif

Classer une publication en deux catégories :

- `organic` : publication normale, nuancée ou prudente.
- `campaign` : publication ressemblant à une campagne coordonnée, répétitive, urgente ou manipulatrice.

## Signaux utilisés

Le modèle regarde :

- le texte de la publication ;
- l'âge du compte ;
- le nombre d'abonnés ;
- le nombre de publications dans les dernières 24h ;
- la présence d'un lien ;
- la similarité avec d'autres messages.

## Structure

```text
desinformation_ml_project/
├── data/
│   └── fake_posts.csv
├── models/
├── notebooks/
│   └── colab_desinformation_detector.ipynb
├── src/
│   ├── train_model.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation locale

```bash
git clone <ton-repo-github>
cd desinformation_ml_project
pip install -r requirements.txt
python src/train_model.py
python src/predict.py
```

## Utilisation dans Google Colab

1. Ouvre `notebooks/colab_desinformation_detector.ipynb`.
2. Exécute les cellules une par une.
3. Modifie les exemples de textes.
4. Ajoute plus de données fictives ou réelles si tu veux améliorer le projet.

## Améliorations possibles

- Ajouter plus de données.
- Ajouter des langues différentes.
- Détecter les hashtags coordonnés.
- Ajouter une visualisation temporelle.
- Utiliser BERT ou CamemBERT pour le français.
- Créer une interface Gradio ou Streamlit.
- Ajouter un système d'explication : pourquoi le modèle pense que c'est une campagne.

## Éthique

Un bon projet AI for Good doit réduire les dommages sans créer de nouveaux risques.
Il faut éviter :

- la surveillance massive ;
- les accusations automatiques ;
- les décisions sans humain ;
- les modèles biaisés contre des groupes politiques, culturels ou linguistiques.

Le bon usage : aider des analystes humains à repérer des signaux suspects, pas remplacer le jugement humain.
