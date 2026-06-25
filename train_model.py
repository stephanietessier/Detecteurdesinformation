"""
Détection simple de campagnes de désinformation avec données fictives.

Objectif pédagogique : entraîner un modèle qui distingue des publications
organiques de publications ressemblant à une campagne coordonnée.

Ce projet n'est PAS un outil de modération automatique prêt pour la production.
Il sert à apprendre : dataset, prétraitement, entraînement, évaluation, prédiction.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fake_posts.csv"
MODEL_PATH = ROOT / "models" / "desinformation_detector.joblib"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge le dataset fictif."""
    df = pd.read_csv(path)
    required = {
        "text",
        "account_age_days",
        "followers",
        "posts_last_24h",
        "has_url",
        "duplicate_similarity",
        "label",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")
    return df


def build_pipeline() -> Pipeline:
    """Crée un pipeline texte + métadonnées."""
    text_features = "text"
    numeric_features = [
        "account_age_days",
        "followers",
        "posts_last_24h",
        "has_url",
        "duplicate_similarity",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(lowercase=True, ngram_range=(1, 2)), text_features),
            ("meta", StandardScaler(), numeric_features),
        ]
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def train_and_evaluate() -> Pipeline:
    df = load_data()
    X = df.drop(columns=["label", "post_id"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print("\n=== Rapport de classification ===")
    print(classification_report(y_test, predictions))

    print("\n=== Matrice de confusion ===")
    print(confusion_matrix(y_test, predictions, labels=["organic", "campaign"]))
    print("Ordre des classes: organic, campaign")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModèle sauvegardé dans: {MODEL_PATH}")
    return pipeline


def predict_examples(pipeline: Pipeline) -> None:
    examples = pd.DataFrame(
        [
            {
                "text": "Copiez ce message partout, ils veulent cacher la vérité demain matin.",
                "account_age_days": 8,
                "followers": 30,
                "posts_last_24h": 80,
                "has_url": 1,
                "duplicate_similarity": 0.95,
            },
            {
                "text": "Je cherche une source indépendante avant de partager cette nouvelle.",
                "account_age_days": 700,
                "followers": 500,
                "posts_last_24h": 2,
                "has_url": 0,
                "duplicate_similarity": 0.04,
            },
        ]
    )

    probs = pipeline.predict_proba(examples)
    classes = list(pipeline.classes_)
    campaign_index = classes.index("campaign")

    print("\n=== Exemples de prédiction ===")
    for i, row in examples.iterrows():
        label = pipeline.predict(examples.iloc[[i]])[0]
        score = probs[i][campaign_index]
        print(f"Texte: {row['text']}")
        print(f"Prédiction: {label} | probabilité campagne: {score:.2%}\n")


if __name__ == "__main__":
    trained_pipeline = train_and_evaluate()
    predict_examples(trained_pipeline)
