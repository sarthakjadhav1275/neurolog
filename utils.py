def get_log_stats(df):

    stats = {
        "Total Logs": len(df),
        "INFO": (df["level"] == "INFO").sum(),
        "WARN": (df["level"] == "WARN").sum(),
        "ERROR": (df["level"] == "ERROR").sum(),
        "CRITICAL": (df["level"] == "CRITICAL").sum()
    }

    return stats
