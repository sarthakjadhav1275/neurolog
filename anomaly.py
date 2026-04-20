import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest


def detect_anomalies(df, contamination: float = 0.10):

    messages = df["message"].astype(str).values

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    embeddings = vectorizer.fit_transform(messages)

    model = IsolationForest(contamination=contamination, random_state=42)
    # Important: avoid converting sparse TF-IDF embeddings to dense arrays.
    # Dense conversion can crash for large logs (OOM).
    # sklearn's IsolationForest validates with accept_sparse=["csc"], so use CSC format.
    preds = model.fit_predict(embeddings.tocsc())

    df["anomaly"] = np.where(preds == -1, "Yes", "No")

    return df
