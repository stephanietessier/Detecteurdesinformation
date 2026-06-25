from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "desinformation_detector.joblib"


def predict_post(text: str, account_age_days: int, followers: int, posts_last_24h: int, has_url: int, duplicate_similarity: float):
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([
        {
            "text": text,
            "account_age_days": account_age_days,
            "followers": followers,
            "posts_last_24h": posts_last_24h,
            "has_url": has_url,
            "duplicate_similarity": duplicate_similarity,
        }
    ])
    prediction = model.predict(row)[0]
    classes = list(model.classes_)
    campaign_score = model.predict_proba(row)[0][classes.index("campaign")]
    return prediction, campaign_score


if __name__ == "__main__":
    label, score = predict_post(
        text="Partagez tous ce message maintenant, les médias cachent la vérité.",
        account_age_days=6,
        followers=20,
        posts_last_24h=75,
        has_url=1,
        duplicate_similarity=0.96,
    )
    print(f"Prédiction: {label}")
    print(f"Score campagne: {score:.2%}")
