import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def cluster_failures(df: pd.DataFrame, anomaly_column: str = "anomaly_ml", max_clusters: int = 3):
    """
    Cluster anomalous log messages using TF-IDF + KMeans.

    - df: DataFrame containing at least 'message' and an anomaly indicator column.
    - anomaly_column: column used to filter anomalies. Can be:
        * boolean (True/False), or
        * string values where "Yes" means anomaly.
    - max_clusters: upper bound on number of clusters (auto-limited by anomaly count).
    """

    if anomaly_column not in df.columns:
        return None

    col = df[anomaly_column]

    if col.dtype == bool:
        mask = col
    else:
        mask = col.astype(str).str.upper().eq("YES")

    failure_df = df[mask].copy()

    if len(failure_df) == 0:
        return None

    messages = failure_df["message"].astype(str).values

    vectorizer = TfidfVectorizer(max_features=1000)
    vectors = vectorizer.fit_transform(messages)

    k = min(max_clusters, len(failure_df))
    if k < 1:
        return None

    model = KMeans(n_clusters=k, random_state=42, n_init="auto")

    failure_df["cluster"] = model.fit_predict(vectors)

    return failure_df
