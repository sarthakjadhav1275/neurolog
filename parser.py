import pandas as pd
import re
from datetime import datetime

timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
level_keywords = ["INFO", "ERROR", "WARN", "DEBUG", "CRITICAL"]


def extract_log_fields(line):

    # Extract timestamp if present
    timestamp_match = re.search(timestamp_pattern, line)
    timestamp = timestamp_match.group(0) if timestamp_match else None

    # Extract level if present
    level = None
    for lvl in level_keywords:
        if lvl in line:
            level = lvl
            break

    # Remaining text as message
    message = line
    if timestamp:
        message = message.replace(timestamp, "")
    if level:
        message = message.replace(level, "")

    return timestamp, level, message.strip()


def read_log_file(file):

    logs = []

    for line in file:
        line = line.decode("utf-8", errors="ignore").strip()

        if not line:
            continue

        ts, lvl, msg = extract_log_fields(line)

        logs.append({
            "timestamp": ts,
            "level": lvl,
            "message": msg
        })

    df = pd.DataFrame(logs)
    return df
