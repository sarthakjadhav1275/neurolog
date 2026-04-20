import pandas as pd

def find_recurring_patterns(df):

    failure_logs = df[df["anomaly"] == "Yes"]

    if len(failure_logs) == 0:
        return None

    pattern_summary = (
        failure_logs["message"]
        .value_counts()
        .reset_index()
    )

    pattern_summary.columns = ["failure_pattern", "occurrences"]

    return pattern_summary

def failure_trend_summary(df):

    trend = (
        df[df["anomaly"] == "Yes"]
        .groupby("level")
        .size()
        .reset_index(name="count")
    )

    return trend
