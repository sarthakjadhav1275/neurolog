from typing import Optional
import streamlit as st
import pandas as pd
import re
from collections import Counter
from anomaly import detect_anomalies as ml_detect_anomalies
from clustering import cluster_failures
from parser import read_log_file
import plotly.express as px
import plotly.graph_objects as go
import os
import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime, timezone, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Additional imports for enhancements
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

st.set_page_config(
    page_title="NeuroLog — Failure Pattern & Log Insight System",
    layout="wide",
)

# App paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

# ======================
# Enhanced Features
# ======================

# Theme Management
def init_theme():
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    
    theme = st.session_state.theme
    
    if theme == "light":
        theme_css = """
        <style>
        :root {
          --nl-bg: #FAFBFC;
          --nl-panel: #FFFFFF;
          --nl-panel-2: #F8FAFC;
          --nl-border: rgba(229, 231, 235, 0.8);
          --nl-text: #1F2937;
          --nl-muted: rgba(107, 114, 128, 0.9);
          --nl-primary: #3B82F6;
          --nl-primary-2: rgba(59, 130, 246, 0.15);
          --nl-primary-hover: #2563EB;
          --nl-accent-yellow: #F59E0B;
          --nl-accent-yellow-soft: #FCD34D;
          --nl-accent-yellow-deep: #D97706;
          --nl-success: #10B981;
          --nl-warning: #F59E0B;
          --nl-error: #EF4444;
          --nl-gradient-start: #3B82F6;
          --nl-gradient-end: #2563EB;
        }
        
        /* Light theme background */
        [data-testid="stAppViewContainer"] {
          background: linear-gradient(135deg, #FAFBFC 0%, #F3F4F6 50%, #E5E7EB 100%);
          color: var(--nl-text);
          transition: all 0.3s ease;
        }
        
        /* Smooth transitions */
        * {
          transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        }
        
        /* Enhanced light theme components */
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
          border-right: 1px solid var(--nl-border);
        }
        
        .neurolog-card {
          background: var(--nl-panel);
          border: 1px solid var(--nl-border);
          border-radius: 16px;
          padding: 20px;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
          transition: all 0.3s ease;
        }
        
        .neurolog-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        </style>
        """
    else:
        theme_css = """
        <style>
        :root {
          --nl-bg: #0F172A;
          --nl-panel: #1E293B;
          --nl-panel-2: #334155;
          --nl-border: rgba(71, 85, 105, 0.6);
          --nl-text: #F1F5F9;
          --nl-muted: rgba(148, 163, 184, 0.9);
          --nl-primary: #60A5FA;
          --nl-primary-2: rgba(96, 165, 250, 0.2);
          --nl-primary-hover: #3B82F6;
          --nl-accent-yellow: #FBBF24;
          --nl-accent-yellow-soft: #FCD34D;
          --nl-accent-yellow-deep: #F59E0B;
          --nl-success: #34D399;
          --nl-warning: #FBBF24;
          --nl-error: #F87171;
          --nl-gradient-start: #60A5FA;
          --nl-gradient-end: #3B82F6;
        }
        
        /* Dark theme background with animated gradient */
        [data-testid="stAppViewContainer"] {
          background: 
            radial-gradient(ellipse at top left, rgba(96, 165, 250, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(251, 191, 36, 0.1) 0%, transparent 50%),
            linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
          color: var(--nl-text);
          transition: all 0.3s ease;
          animation: gradientShift 10s ease infinite;
        }
        
        @keyframes gradientShift {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        
        /* Smooth transitions */
        * {
          transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease, transform 0.2s ease;
        }
        
        /* Enhanced dark theme components */
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
          border-right: 1px solid var(--nl-border);
        }
        
        .neurolog-card {
          background: var(--nl-panel);
          border: 1px solid var(--nl-border);
          border-radius: 16px;
          padding: 20px;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
          transition: all 0.3s ease;
        }
        
        .neurolog-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.3);
        }
        </style>
        """
    
    st.markdown(theme_css, unsafe_allow_html=True)

# Performance: Batch Processing
def parse_logs_batch(text, batch_size=1000, progress_callback=None):
    """Parse logs in batches for better performance with large files"""
    lines = text.splitlines()
    total_lines = len(lines)
    rows = []
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(INFO|WARN|ERROR|CRITICAL)\s+(.*)"
    
    for i in range(0, total_lines, batch_size):
        batch_lines = lines[i:i+batch_size]
        batch_rows = []
        
        for line in batch_lines:
            match = re.match(pattern, line)
            if match:
                batch_rows.append({
                    "timestamp": match.group(1),
                    "level": match.group(2),
                    "message": match.group(3)
                })
        
        rows.extend(batch_rows)
        
        if progress_callback:
            progress = min((i + batch_size) / total_lines, 1.0)
            progress_callback(progress)
    
    return pd.DataFrame(rows)

# Anomaly Severity Scoring
def calculate_anomaly_severity(df, anomaly_column):
    """Calculate severity scores for anomalies"""
    if anomaly_column not in df.columns:
        return df
    
    severity_scores = []
    
    for _, row in df.iterrows():
        score = 0
        message = str(row["message"]).lower()
        level = str(row["level"]).upper()
        
        # Level-based scoring
        level_scores = {"INFO": 1, "WARN": 3, "ERROR": 5, "CRITICAL": 7}
        score += level_scores.get(level, 2)
        
        # Keyword-based scoring
        critical_keywords = ["crash", "fatal", "panic", "exception", "stack trace", "out of memory"]
        high_keywords = ["failed", "timeout", "denied", "blocked", "error"]
        medium_keywords = ["warning", "deprecated", "slow"]
        
        for keyword in critical_keywords:
            if keyword in message:
                score += 3
                break
        
        for keyword in high_keywords:
            if keyword in message:
                score += 2
                break
                
        for keyword in medium_keywords:
            if keyword in message:
                score += 1
                break
        
        # Length-based scoring (longer messages might be more severe)
        if len(message) > 200:
            score += 1
        
        severity_scores.append(min(score, 10))  # Cap at 10
    
    df["anomaly_severity"] = severity_scores
    return df

# Time Series Forecasting
def forecast_error_trends(df, periods=24):
    """Forecast error trends using linear regression"""
    if not ML_AVAILABLE:
        return None, "Machine learning libraries not available"
    
    try:
        # Prepare time series data
        df_time = df.copy()
        df_time["timestamp"] = pd.to_datetime(df_time["timestamp"], errors="coerce")
        df_time = df_time.dropna(subset=["timestamp"])
        
        # Filter for errors and critical
        error_df = df_time[df_time["level"].isin(["ERROR", "CRITICAL"])]
        
        if len(error_df) < 10:
            return None, "Not enough error data for forecasting"
        
        # Create hourly counts
        error_df["hour"] = error_df["timestamp"].dt.floor("h")
        hourly_counts = error_df.groupby("hour").size().reset_index(name="count")
        
        # Prepare for regression
        hourly_counts["hour_numeric"] = range(len(hourly_counts))
        X = hourly_counts[["hour_numeric"]].values
        y = hourly_counts["count"].values
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast
        last_hour = hourly_counts["hour_numeric"].iloc[-1]
        future_hours = [[last_hour + i] for i in range(1, periods + 1)]
        forecast = model.predict(future_hours)
        
        # Create forecast dataframe
        last_timestamp = hourly_counts["hour"].iloc[-1]
        future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, periods + 1)]
        
        forecast_df = pd.DataFrame({
            "timestamp": future_timestamps,
            "forecast_count": forecast,
            "hour_numeric": range(last_hour + 1, last_hour + periods + 1)
        })
        
        return forecast_df, None
        
    except Exception as e:
        return None, f"Forecasting error: {str(e)}"

# Enhanced Export Functions
def export_to_json(df, include_metadata=True):
    """Export dataframe to JSON with enhanced formatting"""
    export_data = {
        "logs": df.to_dict('records'),
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(df)
    }
    
    if include_metadata:
        export_data["metadata"] = {
            "levels": df["level"].value_counts().to_dict(),
            "date_range": {
                "start": df["timestamp"].min() if "timestamp" in df.columns else None,
                "end": df["timestamp"].max() if "timestamp" in df.columns else None
            }
        }
    
    return json.dumps(export_data, indent=2, default=str)

def export_to_pdf(df, title="NeuroLog Report"):
    """Export dataframe to PDF report"""
    if not PDF_AVAILABLE:
        return None, "PDF libraries not available"
    
    try:
        from io import BytesIO
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 12))
        
        # Summary
        summary_data = [
            ['Metric', 'Value'],
            ['Total Logs', str(len(df))],
            ['INFO', str((df['level'] == 'INFO').sum())],
            ['WARN', str((df['level'] == 'WARN').sum())],
            ['ERROR', str((df['level'] == 'ERROR').sum())],
            ['CRITICAL', str((df['level'] == 'CRITICAL').sum())]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 12))
        
        doc.build(story)
        buffer.seek(0)
        
        return buffer.getvalue(), None
        
    except Exception as e:
        return None, f"PDF generation error: {str(e)}"

# Initialize theme
init_theme()

# If you pasted the logo via the Cursor UI, it may be stored under the Cursor project assets.
# This fallback helps during local development.
# If you pasted the logo via Cursor chat/image, it may exist here on your machine.
# This is a local-only fallback; for deployment, put the logo at assets/logo.png.
FALLBACK_CURSOR_LOGO = r"C:\Users\sarth\.cursor\projects\c-Users-sarth-OneDrive-Desktop-AQI-mini-proj-sem-6-NeuroLog-main\assets\c__Users_sarth_AppData_Roaming_Cursor_User_workspaceStorage_66fdbf99bd795b3182fc81aace4e35a7_images_log-file-ff990187-4d7f-417f-aaec-4713198ac858.png"


def _pick_logo_path() -> Optional[str]:
    """Pick a logo from assets folder.

    Priority:
    1) assets/logo.png
    2) any .png in assets folder (prefer names containing 'logo')
    3) fallback to Cursor-stored logo (local dev only)
    """
    try:
        if os.path.exists(LOGO_PATH):
            return LOGO_PATH
        if os.path.isdir(ASSETS_DIR):
            pngs = sorted([f for f in os.listdir(ASSETS_DIR) if f.lower().endswith(".png")])
            if pngs:
                preferred = [p for p in pngs if "logo" in p.lower()]
                chosen = preferred[0] if preferred else pngs[0]
                return os.path.join(ASSETS_DIR, chosen)
        if os.path.exists(FALLBACK_CURSOR_LOGO):
            return FALLBACK_CURSOR_LOGO
    except Exception:
        return None
    return None

# ======================
# Auth (login/register)
# ======================

DB_PATH = os.path.join(PROJECT_DIR, "neurolog_users.sqlite3")


def _utc_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username_norm TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL DEFAULT '',
                pw_salt_hex TEXT NOT NULL,
                pw_hash_hex TEXT NOT NULL,
                created_at_epoch INTEGER NOT NULL,
                last_login_epoch INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS login_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username_norm TEXT NOT NULL,
                attempted_at_epoch INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_locks (
                username_norm TEXT NOT NULL PRIMARY KEY,
                locked_until_epoch INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at_epoch INTEGER NOT NULL,
                log_file_name TEXT,
                log_format TEXT,
                rule_keywords_text TEXT,
                ml_contamination REAL,
                total_logs INTEGER,
                info_count INTEGER,
                warn_count INTEGER,
                error_count INTEGER,
                critical_count INTEGER,
                rule_anomalies INTEGER,
                ml_anomalies INTEGER,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )


def _normalize_username(username: str) -> str:
    return username.strip().lower()


def _hash_password(password: str, *, salt_hex: Optional[str] = None) -> tuple[str, str]:
    if salt_hex is None:
        salt = secrets.token_bytes(16)
        salt_hex = salt.hex()
    else:
        salt = bytes.fromhex(salt_hex)

    iterations = 200_000
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return salt_hex, pw_hash.hex()


def _verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    _, computed_hash_hex = _hash_password(password, salt_hex=salt_hex)
    return hmac.compare_digest(computed_hash_hex, expected_hash_hex)


def _is_locked(username_norm: str) -> tuple[bool, Optional[int]]:
    now = _utc_epoch()
    with _db_connect() as conn:
        row = conn.execute(
            "SELECT locked_until_epoch FROM user_locks WHERE username_norm = ?",
            (username_norm,),
        ).fetchone()
        if not row:
            return False, None
        locked_until = int(row["locked_until_epoch"])
        if locked_until <= now:
            conn.execute("DELETE FROM user_locks WHERE username_norm = ?", (username_norm,))
            return False, None
        return True, locked_until


def _record_failure(username_norm: str) -> None:
    now = _utc_epoch()
    max_failed_attempts = 5
    window_seconds = 10 * 60
    lock_seconds = 60

    with _db_connect() as conn:
        conn.execute(
            "INSERT INTO login_failures (username_norm, attempted_at_epoch) VALUES (?, ?)",
            (username_norm, now),
        )
        cutoff = now - window_seconds
        failed_count = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM login_failures
            WHERE username_norm = ?
              AND attempted_at_epoch >= ?
            """,
            (username_norm, cutoff),
        ).fetchone()["c"]

        if int(failed_count) >= max_failed_attempts:
            locked_until = now + lock_seconds
            conn.execute(
                """
                INSERT INTO user_locks (username_norm, locked_until_epoch)
                VALUES (?, ?)
                ON CONFLICT(username_norm) DO UPDATE SET locked_until_epoch = excluded.locked_until_epoch
                """,
                (username_norm, locked_until),
            )


def _logout_user() -> None:
    for key in ("user_id", "username_norm", "full_name"):
        if key in st.session_state:
            del st.session_state[key]


def _create_user_account(username: str, full_name: str, password: str) -> tuple[bool, str]:
    username_norm = _normalize_username(username)
    if not username_norm:
        return False, "Username cannot be empty."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    created_at_epoch = _utc_epoch()
    salt_hex, hash_hex = _hash_password(password)
    full_name = (full_name or "").strip()

    with _db_connect() as conn:
        try:
            conn.execute(
                """
                INSERT INTO users (
                    username_norm, full_name, pw_salt_hex, pw_hash_hex, created_at_epoch, last_login_epoch
                ) VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (username_norm, full_name, salt_hex, hash_hex, created_at_epoch),
            )
        except sqlite3.IntegrityError:
            return False, "Username already exists. Try another one."

    return True, "Account created successfully."


def _authenticate_user(username: str, password: str) -> tuple[bool, str, Optional[dict]]:
    username_norm = _normalize_username(username)
    if not username_norm:
        return False, "Invalid username or password.", None

    locked, locked_until = _is_locked(username_norm)
    if locked and locked_until is not None:
        seconds_left = max(0, locked_until - _utc_epoch())
        return False, f"Too many failed attempts. Try again in {seconds_left} seconds.", None

    with _db_connect() as conn:
        row = conn.execute(
            "SELECT id, username_norm, full_name, pw_salt_hex, pw_hash_hex FROM users WHERE username_norm = ?",
            (username_norm,),
        ).fetchone()

        # Generic message to reduce user enumeration.
        if not row:
            _record_failure(username_norm)
            return False, "Invalid username or password.", None

        ok = _verify_password(password, row["pw_salt_hex"], row["pw_hash_hex"])
        if not ok:
            _record_failure(username_norm)
            return False, "Invalid username or password.", None

        now = _utc_epoch()
        conn.execute(
            "UPDATE users SET last_login_epoch = ? WHERE id = ?",
            (now, int(row["id"])),
        )
        conn.execute("DELETE FROM user_locks WHERE username_norm = ?", (username_norm,))

    user = {"user_id": int(row["id"]), "username_norm": row["username_norm"], "full_name": row["full_name"]}
    return True, "Login successful.", user


def _get_user_profile(user_id: int) -> Optional[dict]:
    with _db_connect() as conn:
        row = conn.execute(
            "SELECT id, username_norm, full_name, created_at_epoch, last_login_epoch FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        if not row:
            return None
        return dict(row)


def _record_user_activity(
    *,
    user_id: int,
    log_file_name: str,
    log_format: str,
    rule_keywords_text: str,
    ml_contamination: float,
    total_logs: int,
    info_count: int,
    warn_count: int,
    error_count: int,
    critical_count: int,
    rule_anomalies: int,
    ml_anomalies: int,
) -> None:
    created_at_epoch = _utc_epoch()
    with _db_connect() as conn:
        conn.execute(
            """
            INSERT INTO user_activities (
                user_id, created_at_epoch, log_file_name, log_format, rule_keywords_text, ml_contamination,
                total_logs, info_count, warn_count, error_count, critical_count,
                rule_anomalies, ml_anomalies
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(user_id),
                created_at_epoch,
                log_file_name,
                log_format,
                rule_keywords_text,
                float(ml_contamination),
                int(total_logs),
                int(info_count),
                int(warn_count),
                int(error_count),
                int(critical_count),
                int(rule_anomalies),
                int(ml_anomalies),
            ),
        )


def _get_user_activities(user_id: int, limit: int = 20) -> pd.DataFrame:
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT created_at_epoch, log_file_name, log_format,
                   total_logs, info_count, warn_count, error_count, critical_count,
                   rule_anomalies, ml_anomalies
            FROM user_activities
            WHERE user_id = ?
            ORDER BY created_at_epoch DESC
            LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df

    df["when"] = pd.to_datetime(df["created_at_epoch"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    df = df.drop(columns=["created_at_epoch"])
    return df


_init_db()

# ------------------------
# UI polish (no feature changes)
# ------------------------
st.markdown(
    """
<style>
/* Import a modern web font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Apply font globally */
html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
  font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
}

/* Slightly stronger headings */
h1, h2, h3, h4, h5, h6 {
  font-weight: 700 !important;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, var(--nl-primary), var(--nl-accent-yellow));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Hide Streamlit default chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* Hide Streamlit top header bar */
header { visibility: hidden; height: 0; }

/* Remove Streamlit's top decoration / extra bars */
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] {
  display: none !important;
}
[data-testid="stToolbar"] { visibility: hidden !important; }
[data-testid="stAppToolbar"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stTopNav"] { display: none !important; }
/* Streamlit versions vary: hide remaining chrome but keep content */
[data-testid="stAppHeader"] { visibility: hidden !important; height: 0 !important; }
[data-testid="stAppBar"] { visibility: hidden !important; height: 0 !important; }
div[role="banner"] { visibility: hidden !important; height: 0 !important; }

/* Global spacing + responsive page width (fixes "too much empty space") */
.block-container {
  /* extra top padding prevents overlap with Streamlit top bar */
  padding-top: 1.75rem;
  padding-bottom: 2.0rem;
  max-width: 1480px;   /* wider, more dashboard-like */
}

/* On very large screens, use a bit more width */
@media (min-width: 1700px) {
  .block-container { max-width: 1680px; }
}

/* Enhanced animations */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}

@keyframes slideIn {
  from { transform: translateX(-100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}

/* Apply animations to elements */
div[style*="animation: fadeIn"] {
  animation: fadeIn 0.6s ease-out;
}

[data-testid="stMetric"] {
  animation: fadeIn 0.8s ease-out;
}

/* Enhanced File uploader and widget surfaces */
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzone"] section,
[data-testid="stFileUploader"] section {
  background: var(--nl-panel) !important;
  border: 2px solid var(--nl-primary) !important;
  color: var(--nl-text) !important;
  border-radius: 16px !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}

[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploader"]:hover {
  border-color: var(--nl-accent-yellow) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 16px rgba(0,0,0,0.2) !important;
}

[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
  background: linear-gradient(135deg, var(--nl-gradient-start), var(--nl-gradient-end)) !important;
  color: var(--nl-panel) !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  transition: all 0.3s ease !important;
}

/* Enhanced Metric cards */
[data-testid="stMetric"] {
  background: linear-gradient(135deg, var(--nl-panel), var(--nl-panel-2)) !important;
  border: 1px solid var(--nl-border) !important;
  border-radius: 16px !important;
  padding: 20px !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}

[data-testid="stMetric"]:hover {
  transform: translateY(-4px) !important;
  box-shadow: 0 12px 24px rgba(0,0,0,0.2) !important;
  border-color: var(--nl-accent-yellow) !important;
}

/* Enhanced Tabs */
button[data-baseweb="tab"] {
  border-radius: 12px !important;
  padding: 10px 18px !important;
  font-weight: 600 !important;
  border: 2px solid var(--nl-border) !important;
  background: var(--nl-panel) !important;
  color: var(--nl-text) !important;
  margin: 0 2px !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, var(--nl-primary-2), var(--nl-primary)) !important;
  border: 2px solid var(--nl-primary) !important;
  color: white !important;
  box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
  transform: translateY(-1px) !important;
}

button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
  background: var(--nl-panel-2) !important;
  border-color: var(--nl-primary) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
}

/* Enhanced Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
  border-radius: 16px !important;
  border: 2px solid var(--nl-border) !important;
  transition: all 0.3s ease !important;
  background: var(--nl-panel) !important;
}

div[data-baseweb="input"]:hover > div,
div[data-baseweb="select"]:hover > div {
  border-color: var(--nl-primary) !important;
  transform: translateY(-1px) !important;
}

/* Enhanced form styling */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="tag"] {
  background: var(--nl-panel) !important;
  color: var(--nl-text) !important;
  border: 2px solid var(--nl-border) !important;
  transition: all 0.3s ease !important;
}

/* Placeholder and helper text */
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="select"] input::placeholder {
  color: var(--nl-muted) !important;
}

/* Enhanced Dropdown options */
[data-baseweb="menu"] {
  background: var(--nl-panel) !important;
  border: 2px solid var(--nl-primary) !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 16px rgba(0,0,0,0.2) !important;
}

[data-baseweb="menu"] li,
[role="option"] {
  color: var(--nl-text) !important;
  background: transparent !important;
  transition: all 0.2s ease !important;
  border-radius: 8px !important;
  margin: 4px 8px !important;
}

[data-baseweb="menu"] li:hover,
[role="option"]:hover {
  background: var(--nl-primary-2) !important;
  transform: translateX(4px) !important;
}

/* Enhanced Multi-select selected tags */
[data-baseweb="tag"] {
  background: linear-gradient(135deg, var(--nl-accent-yellow-soft), var(--nl-accent-yellow)) !important;
  border: 2px solid var(--nl-accent-yellow-deep) !important;
  color: var(--nl-bg) !important;
  border-radius: 20px !important;
  font-weight: 600 !important;
  transition: all 0.3s ease !important;
}

[data-baseweb="tag"] * {
  color: var(--nl-bg) !important;
}

/* Enhanced Dataframe */
[data-testid="stDataFrame"] {
  border-radius: 16px !important;
  overflow: hidden !important;
  border: 2px solid var(--nl-border) !important;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
  transition: all 0.3s ease !important;
}

[data-testid="stDataFrame"]:hover {
  border-color: var(--nl-primary) !important;
  box-shadow: 0 8px 16px rgba(0,0,0,0.2) !important;
}

/* Enhanced Expanders */
details {
  background: linear-gradient(135deg, var(--nl-panel), var(--nl-panel-2)) !important;
  border: 2px solid var(--nl-border) !important;
  border-radius: 16px !important;
  padding: 16px 20px !important;
  margin: 8px 0 !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}

details:hover {
  border-color: var(--nl-primary) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 16px rgba(0,0,0,0.2) !important;
}

details summary {
  font-weight: 700 !important;
  color: var(--nl-primary) !important;
  font-size: 1.1rem !important;
}

/* Enhanced Buttons */
button[kind="primary"] {
  border-radius: 16px !important;
  font-weight: 700 !important;
  padding: 12px 24px !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
}

/* Primary buttons: Enhanced YELLOW gradient */
button[kind="primary"] {
  background: linear-gradient(
    135deg,
    var(--nl-accent-yellow-soft) 0%,
    var(--nl-accent-yellow) 55%,
    var(--nl-accent-yellow-deep) 100%
  ) !important;
  color: var(--nl-bg) !important;
  border: 2px solid var(--nl-accent-yellow-deep) !important;
  box-shadow: 0 8px 16px rgba(245,158,11,0.3), 0 4px 8px rgba(245,158,11,0.2) !important;
}

button[kind="primary"]:hover {
  transform: translateY(-2px) scale(1.05) !important;
  filter: brightness(1.1) saturate(1.1) !important;
  box-shadow: 0 12px 24px rgba(245,158,11,0.4), 0 6px 12px rgba(245,158,11,0.3) !important;
}

/* Default buttons: Enhanced BLUE gradient */
div.stButton > button:not([kind="primary"]),
[data-testid="stButton"] button:not([kind="primary"]) {
  background: linear-gradient(135deg, var(--nl-gradient-start), var(--nl-gradient-end)) !important;
  color: var(--nl-panel) !important;
  border: 2px solid var(--nl-primary) !important;
  border-radius: 16px !important;
  font-weight: 700 !important;
  padding: 12px 24px !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 8px 16px rgba(59,130,246,0.3), 0 4px 8px rgba(59,130,246,0.2) !important;
}

div.stButton > button:not([kind="primary"]):hover,
[data-testid="stButton"] button:not([kind="primary"]):hover {
  transform: translateY(-2px) scale(1.05) !important;
  filter: brightness(1.1) saturate(1.1) !important;
  box-shadow: 0 12px 24px rgba(59,130,246,0.4), 0 6px 12px rgba(59,130,246,0.3) !important;
}

/* Enhanced interactive polish */
[data-testid="stMetric"] {
  transition: all 0.3s ease !important;
}

[data-testid="stMetric"]:hover {
  transform: translateY(-4px) scale(1.02) !important;
  box-shadow: 0 16px 32px rgba(0,0,0,0.25) !important;
  border-color: var(--nl-accent-yellow) !important;
}

details {
  transition: all 0.3s ease !important;
}

details:hover {
  box-shadow: 0 16px 32px rgba(0,0,0,0.22) !important;
}

details[open] {
  border-color: var(--nl-primary) !important;
  background: linear-gradient(135deg, var(--nl-panel), var(--nl-primary-2)) !important;
}

[data-testid="stDataFrame"] {
  transition: all 0.3s ease !important;
}

[data-testid="stDataFrame"]:hover {
  box-shadow: 0 16px 32px rgba(0,0,0,0.24) !important;
  border-color: var(--nl-accent-yellow) !important;
}

button[data-baseweb="tab"] {
  transition: all 0.3s ease !important;
}

button[data-baseweb="tab"]:hover {
  transform: translateY(-2px) scale(1.02) !important;
}

/* Enhanced captions / secondary text */
[data-testid="stCaptionContainer"] {
  color: var(--nl-muted) !important;
  font-weight: 500 !important;
}

/* Enhanced Markdown links */
a, a:visited {
  color: var(--nl-accent-yellow) !important;
  font-weight: 600 !important;
  text-decoration: none !important;
  transition: all 0.2s ease !important;
}

a:hover {
  color: var(--nl-accent-yellow-deep) !important;
  text-decoration: underline !important;
}

/* Enhanced Progress bars */
[data-testid="stProgressWidget"] {
  background: var(--nl-panel-2) !important;
  border-radius: 20px !important;
  overflow: hidden !important;
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.1) !important;
}

[data-testid="stProgressWidget"] > div {
  background: linear-gradient(90deg, var(--nl-primary), var(--nl-accent-yellow)) !important;
  border-radius: 20px !important;
  transition: all 0.3s ease !important;
}

/* Enhanced Sliders */
[data-testid="stSlider"] > div > div > div {
  background: linear-gradient(90deg, var(--nl-primary), var(--nl-accent-yellow)) !important;
  border-radius: 10px !important;
}

/* Enhanced Selectbox */
[data-testid="stSelectbox"] > div > div {
  background: var(--nl-panel) !important;
  border: 2px solid var(--nl-border) !important;
  border-radius: 16px !important;
  transition: all 0.3s ease !important;
}

[data-testid="stSelectbox"]:hover > div > div {
  border-color: var(--nl-primary) !important;
}

/* Enhanced Text inputs */
[data-testid="stTextInput"] > div > div > input {
  background: var(--nl-panel) !important;
  border: 2px solid var(--nl-border) !important;
  border-radius: 16px !important;
  transition: all 0.3s ease !important;
}

[data-testid="stTextInput"]:focus-within > div > div > input {
  border-color: var(--nl-primary) !important;
  box-shadow: 0 0 0 3px var(--nl-primary-2) !important;
}

/* Enhanced Success/Info/Error messages */
[data-testid="stAlert"] {
  border-radius: 16px !important;
  border: 2px solid !important;
  padding: 16px 20px !important;
  font-weight: 600 !important;
  transition: all 0.3s ease !important;
}

[data-testid="stAlert"][data-testid="stException"] {
  background: linear-gradient(135deg, rgba(248,113,113,0.1), rgba(239,68,68,0.05)) !important;
  border-color: var(--nl-error) !important;
  color: var(--nl-error) !important;
}

[data-testid="stAlert"][data-testid="stInfo"] {
  background: linear-gradient(135deg, rgba(96,165,250,0.1), rgba(59,130,246,0.05)) !important;
  border-color: var(--nl-primary) !important;
  color: var(--nl-primary) !important;
}

[data-testid="stAlert"][data-testid="stSuccess"] {
  background: linear-gradient(135deg, rgba(52,211,153,0.1), rgba(16,185,129,0.05)) !important;
  border-color: var(--nl-success) !important;
  color: var(--nl-success) !important;
}

/* Enhanced Loading spinner */
[data-testid="stSpinner"] > div {
  border: 4px solid var(--nl-panel-2) !important;
  border-top: 4px solid var(--nl-primary) !important;
  border-radius: 50% !important;
  animation: spin 1s linear infinite !important;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
    """,
    unsafe_allow_html=True,
)

# Clean header without complex styling
col_logo, col_title, col_theme = st.columns([1, 8, 1], vertical_alignment="center")
with col_logo:
    logo_path = _pick_logo_path()
    if logo_path and os.path.exists(logo_path):
        st.image(logo_path, width=72)
    else:
        st.markdown(
            '<div style="width:72px;height:72px;border-radius:16px;background:linear-gradient(135deg, var(--nl-primary), var(--nl-accent-yellow));'
            'display:flex;align-items:center;justify-content:center;'
            'font-weight:800;font-size:24px;color:white;">NL</div>',
            unsafe_allow_html=True,
        )
with col_title:
    st.markdown(
        f"""
<div style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, var(--nl-primary), var(--nl-accent-yellow)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
  NeuroLog
</div>
<div style="margin-top: 0.5rem; color: var(--nl-muted); font-weight: 500; font-size: 1.1rem;">
  Failure Pattern & Log Insight Dashboard
</div>
        """,
        unsafe_allow_html=True,
    )
with col_theme:
    # Enhanced theme toggle button
    theme_icon = "☀️" if st.session_state.theme == "light" else "🌙"
    
    if st.button(theme_icon, key="theme_toggle", help="Switch between light and dark themes"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        init_theme()
        st.rerun()
    
    st.markdown(f"""
<style>
div[data-testid="stButton"] > button[kind="default"][key="theme_toggle"] {{
    background: linear-gradient(135deg, var(--nl-primary), var(--nl-accent-yellow)) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 60px !important;
    height: 60px !important;
    font-size: 24px !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    transition: all 0.3s ease !important;
}}
div[data-testid="stButton"] > button[kind="default"][key="theme_toggle"]:hover {{
    transform: scale(1.1) rotate(180deg) !important;
    box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
}}
</style>
    """, unsafe_allow_html=True)
st.caption("🚀 Upload logs → 🔍 detect anomalies → 📊 discover patterns → 📄 export insights", unsafe_allow_html=True)

# Clean feature indicators
feature_indicators = st.columns(6)
features = [
    ("🚀", "Performance", "Batch Processing"),
    ("🎨", "Themes", "Dark/Light"),
    ("📡", "Real-time", "Live Streaming"),
    ("📊", "Forecast", "ML Predictions"),
    ("📄", "Export", "PDF/JSON"),
    ("🔌", "API", "RESTful")
]

for i, (icon, title, desc) in enumerate(features):
    with feature_indicators[i]:
        st.markdown(f"""
<div style="text-align: center; padding: 16px; margin: 4px;">
    <div style="font-size: 1.8rem; margin-bottom: 6px;">{icon}</div>
    <div style="font-weight: 600; color: var(--nl-primary); margin-bottom: 2px; font-size: 0.9rem;">{title}</div>
    <div style="font-size: 0.75rem; color: var(--nl-muted);">{desc}</div>
</div>
        """, unsafe_allow_html=True)

st.markdown(
    """
**How to use NeuroLog**

1. Click **Log Analyzer**.
2. Upload a log file and tune anomaly settings.
3. Explore the results in the tabs below, then export reports.
    """
)

st.divider()

# ------------------------
# Navigation (no left sidebar)
# ------------------------
if "page" not in st.session_state:
    user_logged_in = bool(st.session_state.get("user_id"))
    st.session_state["page"] = "home" if user_logged_in else "login"

page = st.session_state.get("page", "home" if bool(st.session_state.get("user_id")) else "login")

if page in ("analyzer", "history") and not st.session_state.get("user_id"):
    st.session_state["page"] = "login"
    page = "login"

# NOTE: No top navigation buttons here.
# Home page will have a single centered CTA button to avoid Streamlit duplicate-element errors.

# ----------------------------------
# Function: Parse Logs (Enhanced with Batch Processing)
# ----------------------------------
def parse_logs(text, use_batch=True, batch_size=1000):

    if use_batch and len(text.splitlines()) > batch_size:
        # Use batch processing for large files
        progress_bar = st.progress(0, text="Parsing logs...")
        
        def progress_callback(progress):
            progress_bar.progress(progress, text=f"Parsing logs... {progress:.1%}")
        
        df = parse_logs_batch(text, batch_size=batch_size, progress_callback=progress_callback)
        progress_bar.empty()
        return df
    else:
        # Use original parsing for smaller files
        rows = []
        pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(INFO|WARN|ERROR|CRITICAL)\s+(.*)"

        for line in text.splitlines():
            match = re.match(pattern, line)
            if match:
                rows.append({
                    "timestamp": match.group(1),
                    "level": match.group(2),
                    "message": match.group(3)
                })

        return pd.DataFrame(rows)


# ----------------------------------
# Function: Detect Anomalies (Rule-based)
# ----------------------------------
def detect_anomalies_rule_based(df, keywords_text: str):

    if not keywords_text.strip():
        keywords = ["failed", "timeout", "crashed", "denied", "blocked", "error", "exception"]
    else:
        keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]

    df["anomaly_rule"] = df["message"].str.contains(
        "|".join(keywords),
        case=False,
        regex=True
    )

    return df


# ----------------------------------
# Function: Apply ML-based Anomaly Detection
# ----------------------------------
def apply_anomaly_detectors(df, rule_keywords_text: str, ml_contamination: float):

    df = detect_anomalies_rule_based(df, rule_keywords_text)

    ml_df = ml_detect_anomalies(df.copy(), contamination=ml_contamination)

    df["anomaly_ml"] = ml_df["anomaly"].astype(str).str.upper().eq("YES")

    return df


# ----------------------------------
# Function: Recurring Failure Patterns
# ----------------------------------
def recurring_failures(df, anomaly_column):

    failure_msgs = df[df[anomaly_column]]["message"].tolist()
    counter = Counter(failure_msgs)

    return pd.DataFrame(counter.items(), columns=["failure_pattern", "occurrences"])


# ----------------------------------
# Function: Failure Trend Summary
# ----------------------------------
def failure_trend(df):
    return df.groupby("level").size().reset_index(name="count")


# ----------------------------------
# Function: Service / Component Extraction
# ----------------------------------
def infer_service_from_message(message: str):
    if not isinstance(message, str):
        return None

    # Look for explicit service=NAME pattern
    m = re.search(r"service=([A-Za-z0-9_-]+)", message)
    if m:
        return m.group(1)

    # Look for tokens like auth-service, payment-service, api-gateway
    tokens = re.findall(r"[A-Za-z0-9_-]+", message)
    for token in tokens:
        if token.endswith(("-service", "-gateway")):
            return token

    return None


def add_service_column(df):
    if "service" in df.columns:
        return df
    df["service"] = df["message"].astype(str).apply(infer_service_from_message)
    return df


# ----------------------------------
# Function: Correlated Failure Pairs
# ----------------------------------
def compute_failure_correlations(df, anomaly_column, window_minutes: int = 5):
    if anomaly_column not in df.columns:
        return None

    df_corr = df.copy()
    df_corr["timestamp"] = pd.to_datetime(df_corr["timestamp"], errors="coerce")
    df_corr = df_corr.dropna(subset=["timestamp"])

    if df_corr.empty:
        return None

    col = df_corr[anomaly_column]
    if col.dtype == bool:
        mask = col
    else:
        mask = col.astype(str).str.upper().eq("YES")

    df_corr = df_corr[mask & df_corr["level"].isin(["ERROR", "CRITICAL"])].sort_values("timestamp")

    if len(df_corr) < 2:
        return None

    pair_counts = Counter()
    rows = df_corr[["timestamp", "message"]].values

    for i in range(len(rows) - 1):
        t1, m1 = rows[i]
        t2, m2 = rows[i + 1]
        if (t2 - t1).total_seconds() <= window_minutes * 60:
            pair_counts[(m1, m2)] += 1

    if not pair_counts:
        return None

    data = [
        {"from_message": pair[0], "to_message": pair[1], "count": count}
        for pair, count in pair_counts.items()
    ]

    return pd.DataFrame(data).sort_values("count", ascending=False)


# ======================================================
# HOME PAGE CONTENT
# ======================================================

if page == "login":
    st.subheader("🔐 Login")
    if st.session_state.get("user_id"):
        st.info("You are already logged in.")
    else:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            ok, msg, user = _authenticate_user(username, password)
            if ok and user is not None:
                st.session_state["user_id"] = user["user_id"]
                st.session_state["username_norm"] = user["username_norm"]
                st.session_state["full_name"] = user["full_name"]
                st.session_state["page"] = "home"
                st.rerun()
            st.error(msg)

        if st.button("Create account", width='stretch', key="create_account_btn"):
            st.session_state["page"] = "register"
            st.rerun()

elif page == "register":
    st.subheader("🧾 Create Account")
    if st.session_state.get("user_id"):
        st.info("You are already logged in.")
    else:
        with st.form("register_form", clear_on_submit=False):
            username = st.text_input("Username")
            full_name = st.text_input("Full name (optional)")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account")

        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, msg = _create_user_account(username, full_name, password)
                if not ok:
                    st.error(msg)
                else:
                    ok2, msg2, user = _authenticate_user(username, password)
                    if ok2 and user is not None:
                        st.session_state["user_id"] = user["user_id"]
                        st.session_state["username_norm"] = user["username_norm"]
                        st.session_state["full_name"] = user["full_name"]
                        st.session_state["page"] = "home"
                        st.rerun()
                    st.error(msg2)

        if st.button("Back to login", width='stretch', key="back_to_login_btn"):
            st.session_state["page"] = "login"
            st.rerun()

elif page == "history":
    # Add back button at the top
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("Back to Home", width='stretch', key="history_back_btn"):
            st.session_state["page"] = "home"
            st.rerun()
    
    st.subheader("My Activity History")
    if not st.session_state.get("user_id"):
        st.warning("Please login first.")
        st.session_state["page"] = "login"
        st.rerun()

    profile = _get_user_profile(int(st.session_state["user_id"]))
    if not profile:
        st.warning("Could not load your profile. Please login again.")
        st.session_state["page"] = "login"
        st.rerun()

    display_name = profile.get("full_name") or profile.get("username_norm")
    st.success(f"Logged in as {display_name}")

    col_left, col_right = st.columns([6, 1])
    with col_right:
        if st.button("Logout", width='stretch', key="history_logout_btn"):
            _logout_user()
            st.session_state["page"] = "login"
            st.rerun()

    acts = _get_user_activities(int(st.session_state["user_id"]), limit=20)
    if acts.empty:
        st.info("No history yet. Upload a log in the Analyzer.")
    else:
        st.dataframe(acts, width='stretch')

if page == "home":
    st.markdown(
        """
<div style="text-align: center; padding: 32px; margin: 20px 0; background: var(--nl-panel); border: 1px solid var(--nl-border); border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;">
  <div style="font-size: 1.3rem; font-weight: 700; color: var(--nl-primary); margin-bottom: 16px; animation: fadeIn 0.8s ease-out;">
    🎉 Welcome to NeuroLog
  </div>
  <div style="margin-top: 0.5rem; color: var(--nl-muted); line-height: 1.6; font-size: 1.1rem; animation: fadeIn 1s ease-out;">
    Transform raw log files into <b style="color: var(--nl-accent-yellow);">actionable insights</b> with our advanced AI-powered analytics:
    <br>• <span style="color: var(--nl-success);">✓</span> Smart anomaly detection
    <br>• <span style="color: var(--nl-success);">✓</span> Recurring failure patterns
    <br>• <span style="color: var(--nl-success);">✓</span> ML-powered clustering
    <br>• <span style="color: var(--nl-success);">✓</span> Beautiful exportable reports
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Enhanced action buttons with better styling
    st.markdown('<div style="display: flex; justify-content: center; gap: 20px; margin: 30px 0;">', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([2, 3, 2], vertical_alignment="center")
    
    with c1:
        if st.session_state.get("user_id"):
            st.markdown("""
<div style="text-align: center; padding: 20px; background: var(--nl-panel); border: 1px solid var(--nl-border); border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;">
    <div style="font-size: 2rem; margin-bottom: 12px;">📊</div>
    <div style="font-weight: 600; color: var(--nl-primary); margin-bottom: 8px;">My History</div>
    <div style="font-size: 0.9rem; color: var(--nl-muted); margin-bottom: 16px;">View your past analyses</div>
</div>
            """, unsafe_allow_html=True)
            if st.button("📊 My History", width='stretch', key="home_history_btn"):
                st.session_state["page"] = "history"
                st.rerun()

    with c2:
        st.markdown("""
<div style="text-align: center; padding: 24px; background: linear-gradient(135deg, var(--nl-primary-2), var(--nl-accent-yellow-soft)); border: 2px solid var(--nl-accent-yellow); border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;">
    <div style="font-size: 2.5rem; margin-bottom: 16px; animation: pulse 2s infinite;">🚀</div>
    <div style="font-weight: 700; color: var(--nl-primary); margin-bottom: 12px; font-size: 1.2rem;">Start Analysis</div>
    <div style="font-size: 1rem; color: var(--nl-muted); margin-bottom: 20px;">Upload logs and discover insights</div>
</div>
            """, unsafe_allow_html=True)
        if st.button("🚀 Log Analyzer", type="primary", width='stretch', key="home_analyzer_btn"):
            st.session_state["page"] = "analyzer" if st.session_state.get("user_id") else "login"
            st.rerun()

    with c3:
        if st.session_state.get("user_id"):
            st.markdown("""
<div style="text-align: center; padding: 20px; background: var(--nl-panel); border: 1px solid var(--nl-border); border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;">
    <div style="font-size: 2rem; margin-bottom: 12px;">👋</div>
    <div style="font-weight: 600; color: var(--nl-error); margin-bottom: 8px;">Logout</div>
    <div style="font-size: 0.9rem; color: var(--nl-muted); margin-bottom: 16px;">Sign out safely</div>
</div>
            """, unsafe_allow_html=True)
            if st.button("👋 Logout", width='stretch', key="home_logout_btn"):
                _logout_user()
                st.session_state["page"] = "login"
                st.rerun()
        else:
            st.markdown("""
<div style="text-align: center; padding: 20px; background: var(--nl-panel); border: 1px solid var(--nl-border); border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; opacity: 0.6;">
    <div style="font-size: 2rem; margin-bottom: 12px;">🔒</div>
    <div style="font-weight: 600; color: var(--nl-muted); margin-bottom: 8px;">Login Required</div>
    <div style="font-size: 0.9rem; color: var(--nl-muted);">Sign in to continue</div>
</div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Enhanced feature showcase
    st.markdown('<div style="margin: 40px 0;">', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: var(--nl-primary); margin-bottom: 30px;">🌟 How NeuroLog Works</h2>', unsafe_allow_html=True)
    
    # Feature steps with animations
    steps = [
        ("📁", "Upload Logs", "Drag & drop your .log or .txt files"),
        ("🔍", "Smart Analysis", "AI-powered anomaly detection"),
        ("🧩", "Pattern Discovery", "Find recurring failure patterns"),
        ("📊", "Visual Insights", "Interactive charts and metrics"),
        ("📄", "Export Reports", "Download PDF, JSON, CSV reports")
    ]
    
    cols = st.columns(5)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
<div style="text-align: center; padding: 24px 16px; margin: 8px; background: var(--nl-panel); border: 1px solid var(--nl-border); border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; animation: fadeIn {0.6 + i*0.2}s ease-out;">
    <div style="font-size: 2.5rem; margin-bottom: 16px;">{icon}</div>
    <div style="font-weight: 700; color: var(--nl-primary); margin-bottom: 8px; font-size: 1rem;">{title}</div>
    <div style="font-size: 0.85rem; color: var(--nl-muted); line-height: 1.4;">{desc}</div>
</div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced use cases section
    st.markdown('<div style="margin: 40px 0;">', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: var(--nl-primary); margin-bottom: 30px;">🎯 Perfect For</h2>', unsafe_allow_html=True)
    
    use_cases = [
        ("🔧", "DevOps Teams", "Debug server issues faster"),
        ("🏢", "IT Operations", "Monitor system health"),
        ("📚", "Development", "Improve code quality"),
        ("📊", "Analytics", "Data-driven insights")
    ]
    
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(use_cases):
        with cols[i]:
            st.markdown(f"""
<div style="text-align: center; padding: 20px; margin: 8px; background: linear-gradient(135deg, var(--nl-panel), var(--nl-panel-2)); border: 1px solid var(--nl-border); border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;">
    <div style="font-size: 2rem; margin-bottom: 12px;">{icon}</div>
    <div style="font-weight: 600; color: var(--nl-accent-yellow); margin-bottom: 8px;">{title}</div>
    <div style="font-size: 0.9rem; color: var(--nl-muted);">{desc}</div>
</div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced call-to-action
    st.markdown("""
<div style="text-align: center; padding: 32px; margin: 40px 0; background: linear-gradient(135deg, var(--nl-primary-2), var(--nl-accent-yellow-soft)); border: 2px solid var(--nl-accent-yellow); border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;">
    <div style="font-size: 2rem; margin-bottom: 16px; animation: pulse 2s infinite;">🎉</div>
    <div style="font-weight: 700; color: var(--nl-primary); margin-bottom: 16px; font-size: 1.3rem;">Ready to Transform Your Logs?</div>
    <div style="font-size: 1.1rem; color: var(--nl-muted); margin-bottom: 24px;">Join thousands of developers using NeuroLog for smarter log analysis</div>
</div>
    """, unsafe_allow_html=True)

# ======================================================
# PROCESS FILE (LOG ANALYZER)
# ======================================================

if page == "analyzer":
    if not st.session_state.get("user_id"):
        st.warning("Please login to use the Log Analyzer.")
        st.session_state["page"] = "login"
        st.rerun()

    top_back_left, top_back_mid, top_back_right = st.columns([2, 3, 2], vertical_alignment="center")
    with top_back_left:
        if st.button("← Back to Home", width='stretch', key="analyzer_back_btn"):
            st.session_state["page"] = "home"
            st.rerun()

    st.subheader("📂 Upload & Settings")
    with st.expander("Upload log file and configure detection", expanded=True):
        col_u1, col_u2 = st.columns([2, 3], vertical_alignment="top")

        with col_u1:
            uploaded_file = st.file_uploader(
                "Upload a .log or .txt file",
                type=["log", "txt"],
                help="Supported formats: .log and .txt with timestamps.",
            )
            log_format = st.selectbox(
                "Log format",
                [
                    "Strict (YYYY-MM-DD HH:MM:SS LEVEL message)",
                    "Flexible (timestamp anywhere + level keyword)",
                ],
                help="Choose how your log file is structured. Use 'Strict' for standard timestamp + level lines.",
            )

        with col_u2:
            default_keywords = "failed, timeout, crashed, denied, blocked, error, exception"
            rule_keywords_text = st.text_input(
                "Rule-based anomaly keywords (comma separated)",
                value=default_keywords,
                help="Customize which words indicate potential failures in log messages.",
            )

            ml_contamination = st.slider(
                "ML anomaly contamination (approx. fraction of anomalies)",
                min_value=0.01,
                max_value=0.30,
                value=0.10,
                step=0.01,
                help="Higher values mark more points as anomalies; lower values are stricter.",
            )

    if not uploaded_file:
        st.info("Upload a log file above to start analysis.")
        st.stop()

    if log_format.startswith("Strict"):
        log_text = uploaded_file.read().decode("utf-8", errors="ignore")
        df = parse_logs(log_text, use_batch=True, batch_size=1000)
    else:
        uploaded_file.seek(0)
        df = read_log_file(uploaded_file)

    if df.empty:
        st.error("❌ Could not read logs — ensure logs follow proper timestamp format.")
        st.stop()

    # Apply enhanced anomaly detection
    df = apply_anomaly_detectors(df, rule_keywords_text, ml_contamination)
    df = add_service_column(df)
    
    # Calculate anomaly severity scores
    df = calculate_anomaly_severity(df, "anomaly_ml" if "anomaly_ml" in df.columns else "anomaly_rule")

    st.success("✔ Log file processed successfully")
    st.info(f"📊 Processed **{len(df)}** log entries with enhanced analysis")

    # Persist a summary of this run to the logged-in user's activity history.
    # (We use a signature to avoid duplicate inserts caused by Streamlit reruns.)
    if st.session_state.get("user_id"):
        user_id = int(st.session_state["user_id"])
        total_logs = int(len(df))
        info_count = int((df["level"] == "INFO").sum()) if "level" in df.columns else 0
        warn_count = int((df["level"] == "WARN").sum()) if "level" in df.columns else 0
        error_count = int((df["level"] == "ERROR").sum()) if "level" in df.columns else 0
        critical_count = int((df["level"] == "CRITICAL").sum()) if "level" in df.columns else 0
        rule_anomalies = int(df["anomaly_rule"].sum()) if "anomaly_rule" in df.columns else 0
        ml_anomalies = int(df["anomaly_ml"].sum()) if "anomaly_ml" in df.columns else 0

        ts_series = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df.columns else pd.Series([])
        ts_min = str(ts_series.min()) if not ts_series.empty else ""
        ts_max = str(ts_series.max()) if not ts_series.empty else ""

        signature_material = (
            f"{getattr(uploaded_file, 'name', 'uploaded_log')}|{log_format}|{rule_keywords_text}|"
            f"{ml_contamination}|{total_logs}|{ts_min}|{ts_max}|{rule_anomalies}|{ml_anomalies}"
        )
        activity_signature = hashlib.sha256(signature_material.encode("utf-8")).hexdigest()

        if st.session_state.get("last_activity_signature") != activity_signature:
            _record_user_activity(
                user_id=user_id,
                log_file_name=getattr(uploaded_file, "name", "uploaded_log"),
                log_format=log_format,
                rule_keywords_text=rule_keywords_text,
                ml_contamination=float(ml_contamination),
                total_logs=total_logs,
                info_count=info_count,
                warn_count=warn_count,
                error_count=error_count,
                critical_count=critical_count,
                rule_anomalies=rule_anomalies,
                ml_anomalies=ml_anomalies,
            )
            st.session_state["last_activity_signature"] = activity_signature

    # ------------------------
    # Global interactive filters (apply to all tabs)
    # ------------------------
    st.subheader("🔎 Global Filters")
    with st.expander("Filter logs across the entire dashboard", expanded=True):
        df_filtered = df.copy()

        # Timestamp range filter
        df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"], errors="coerce")
        has_time = df_filtered["timestamp"].notna().any()

        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 3])

        with col_f1:
            selected_levels = st.multiselect(
                "Levels",
                options=["INFO", "WARN", "ERROR", "CRITICAL"],
                default=["INFO", "WARN", "ERROR", "CRITICAL"],
                help="Show only selected log levels.",
            )

        with col_f2:
            service_options = (
                sorted([s for s in df_filtered["service"].dropna().unique().tolist() if str(s).strip()])
                if "service" in df_filtered.columns
                else []
            )
            selected_services = st.multiselect(
                "Service / Component",
                options=service_options,
                default=[],
                help="Optional. Filter by inferred service/component from messages.",
            )

        with col_f3:
            keyword_query = st.text_input(
                "Search in message",
                value="",
                help="Case-insensitive contains match on the message text.",
            )

        with col_f4:
            if has_time:
                min_ts = df_filtered["timestamp"].min()
                max_ts = df_filtered["timestamp"].max()
                time_range = st.slider(
                    "Time range",
                    min_value=min_ts.to_pydatetime(),
                    max_value=max_ts.to_pydatetime(),
                    value=(min_ts.to_pydatetime(), max_ts.to_pydatetime()),
                    help="Filter logs within a timestamp window.",
                )
            else:
                time_range = None
                st.info("No valid timestamps detected for time filtering.")

        # Apply filters
        if selected_levels:
            df_filtered = df_filtered[df_filtered["level"].isin(selected_levels)]

        if selected_services:
            df_filtered = df_filtered[df_filtered["service"].isin(selected_services)]

        if keyword_query.strip():
            df_filtered = df_filtered[
                df_filtered["message"].astype(str).str.contains(keyword_query.strip(), case=False, regex=False)
            ]

        if has_time and time_range is not None:
            start_ts, end_ts = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
            df_filtered = df_filtered[(df_filtered["timestamp"] >= start_ts) & (df_filtered["timestamp"] <= end_ts)]

        st.caption(f"Showing **{len(df_filtered)}** of **{len(df)}** logs after filters.")

    overview_tab, anomalies_tab, clusters_tab, patterns_tab, summary_tab, evaluation_tab, live_tab, forecast_tab, severity_tab = st.tabs(
        ["Overview", "Anomalies", "Clusters", "Patterns", "Summary", "Evaluation", "Live Stream", "🔮 Forecast", "⚡ Severity"]
    )

    with overview_tab:
        st.subheader("📑 Structured Log View")
        with st.expander("Show structured logs table", expanded=True):
            st.dataframe(df_filtered, width='stretch')
        st.info(f"Total Logs Loaded: **{len(df)}** | After filters: **{len(df_filtered)}**")

        st.subheader("📊 Log Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Logs (filtered)", len(df_filtered))
        col2.metric("INFO", (df_filtered.level == "INFO").sum())
        col3.metric("WARN", (df_filtered.level == "WARN").sum())
        col4.metric("ERROR", (df_filtered.level == "ERROR").sum())
        col5.metric("CRITICAL", (df_filtered.level == "CRITICAL").sum())

        st.subheader("📈 Failure Trend Summary")
        trend_df = failure_trend(df_filtered)
        st.dataframe(trend_df, width='stretch')
        if not trend_df.empty:
            pie_fig = px.pie(
                trend_df,
                names="level",
                values="count",
                title="Failure Trend Distribution by Level",
            )
            st.plotly_chart(pie_fig, width='stretch')

        if "service" in df_filtered.columns and df_filtered["service"].notna().any():
            st.subheader("🧩 Logs per Service / Component")
            service_counts = (
                df_filtered.dropna(subset=["service"])
                .groupby("service")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            st.bar_chart(service_counts, x="service", y="count")

    with anomalies_tab:
        st.subheader("🔍 Anomaly Detection")

        if "anomaly_method" not in st.session_state:
            st.session_state["anomaly_method"] = "ML (Isolation Forest TF-IDF)"

        anomaly_method = st.radio(
            "Choose anomaly detection method",
            ["Rule-based (keywords)", "ML (Isolation Forest TF-IDF)"],
            horizontal=True,
            index=0 if st.session_state["anomaly_method"].startswith("Rule") else 1,
        )
        st.session_state["anomaly_method"] = anomaly_method

        if anomaly_method.startswith("Rule"):
            anomaly_col = "anomaly_rule"
        else:
            anomaly_col = "anomaly_ml"

        anomaly_df = df_filtered[df_filtered[anomaly_col]]

        total_logs = len(df_filtered)
        total_anomalies = len(anomaly_df)
        anomaly_ratio = (total_anomalies / total_logs * 100.0) if total_logs else 0.0

        col_a1, col_a2, col_a3 = st.columns(3)
        col_a1.metric("Total Logs", total_logs)
        col_a2.metric("Detected Anomalies", total_anomalies)
        col_a3.metric("Anomaly Rate", f"{anomaly_ratio:.1f}%")

        with st.expander("View anomalous log entries", expanded=True):
            st.dataframe(anomaly_df, width='stretch', height=350)

        st.warning(
            f"Detected **{total_anomalies}** abnormal / unusual log entries using **{anomaly_method}**"
        )

        st.subheader("🗂 Filter Logs")
        filter_choice = st.selectbox(
            "Select view",
            ["All Logs", "Only Anomalies", "Only Critical"]
        )
        if filter_choice == "Only Anomalies":
            filtered = df_filtered[df_filtered[anomaly_col]]
        elif filter_choice == "Only Critical":
            filtered = df_filtered[df_filtered["level"] == "CRITICAL"]
        else:
            filtered = df_filtered
        st.dataframe(filtered, width='stretch', height=400)

    with clusters_tab:
        st.subheader("🧩 Failure Pattern Clustering (KMeans)")

        # Reuse anomaly method selection from anomalies_tab (persisted in session_state)
        cluster_anomaly_col = "anomaly_rule" if st.session_state.get("anomaly_method", "").startswith("Rule") else "anomaly_ml"
        cluster_anomaly_col = cluster_anomaly_col if cluster_anomaly_col in df_filtered.columns else "anomaly_rule"

        max_k = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=3, step=1)
        clustered_df = cluster_failures(df_filtered, anomaly_column=cluster_anomaly_col, max_clusters=max_k)

        if clustered_df is None:
            st.info("Not enough anomalous logs to perform clustering.")
        else:
            cluster_counts = (
                clustered_df.groupby("cluster")
                .size()
                .reset_index(name="count")
                .sort_values("cluster")
            )

            st.plotly_chart(
                px.bar(cluster_counts, x="cluster", y="count", title="Cluster sizes"),
                width='stretch',
            )

            # Fixed the lambda function issue
            cluster_options = []
            for _, row in cluster_counts.iterrows():
                cluster_id = row["cluster"]
                count = int(row["count"])
                cluster_options.append(f"Cluster {cluster_id} (n={count})")

            selected_cluster_idx = st.selectbox(
                "View cluster details",
                options=range(len(cluster_options)),
                format_func=lambda i: cluster_options[i]
            )
            
            selected_cluster = cluster_counts.iloc[selected_cluster_idx]["cluster"]

            cluster_view = clustered_df[clustered_df["cluster"] == selected_cluster][
                ["timestamp", "level", "message", "cluster"]
            ]
            st.dataframe(cluster_view, width='stretch', height=350)

            st.info(
                f"Identified **{len(cluster_counts)}** clusters over "
                f"**{len(clustered_df)}** anomalous logs (using {cluster_anomaly_col})."
            )

    with patterns_tab:
        st.subheader("♻ Recurring Failure Patterns (based on anomalies)")

        # Use ML anomalies for pattern mining when available
        pattern_anomaly_col = "anomaly_rule" if st.session_state.get("anomaly_method", "").startswith("Rule") else "anomaly_ml"
        pattern_anomaly_col = pattern_anomaly_col if pattern_anomaly_col in df_filtered.columns else "anomaly_rule"

        rec_df = recurring_failures(df_filtered, pattern_anomaly_col).sort_values("occurrences", ascending=False)
        with st.expander("View recurring patterns table", expanded=True):
            st.dataframe(rec_df, width='stretch', height=350)

        if not rec_df.empty:
            top_n = st.slider("Top N patterns", min_value=5, max_value=50, value=10, step=5)
            top_df = rec_df.head(int(top_n))
            st.plotly_chart(
                px.bar(top_df, x="occurrences", y="failure_pattern", orientation="h", title="Most frequent patterns"),
                width='stretch',
            )

        st.subheader("⏱ Time-based Error Trend")
        df_time = df_filtered.copy()
        df_time["timestamp"] = pd.to_datetime(df_time["timestamp"], errors="coerce")
        df_time = df_time.dropna(subset=["timestamp"])
        error_mask = df_time["level"].isin(["ERROR", "CRITICAL"])
        if error_mask.any():
            df_errors = df_time[error_mask].set_index("timestamp").resample("5min").size().reset_index(name="error_count")
            st.line_chart(df_errors, x="timestamp", y="error_count")
        else:
            st.info("No ERROR or CRITICAL entries available to build a time-series trend.")

        st.subheader("🔗 Correlated Failure Pairs (within 5 minutes)")
        corr_df = compute_failure_correlations(df, pattern_anomaly_col, window_minutes=5)
        if corr_df is not None and not corr_df.empty:
            st.dataframe(corr_df.head(10), width='stretch')
        else:
            st.info("Not enough anomalous failures to compute correlated pairs.")

    with summary_tab:
        st.subheader("📝 Summary Report")

        total_logs = len(df)
        info_count = (df["level"] == "INFO").sum()
        warn_count = (df["level"] == "WARN").sum()
        error_count = (df["level"] == "ERROR").sum()
        critical_count = (df["level"] == "CRITICAL").sum()

        rule_anomalies = df["anomaly_rule"].sum() if "anomaly_rule" in df.columns else 0
        ml_anomalies = df["anomaly_ml"].sum() if "anomaly_ml" in df.columns else 0

        ml_rec_df = recurring_failures(df, "anomaly_ml") if "anomaly_ml" in df.columns else pd.DataFrame()
        if not ml_rec_df.empty:
            top_pattern = ml_rec_df.iloc[0]["failure_pattern"]
            top_pattern_count = int(ml_rec_df.iloc[0]["occurrences"])
            top_pattern_text = f"Most frequent ML anomaly pattern: **\"{top_pattern}\"** (seen **{top_pattern_count}** times)"
        else:
            top_pattern_text = "No recurring ML anomaly patterns detected."

        ml_clustered_df = cluster_failures(df, anomaly_column="anomaly_ml", max_clusters=3) if "anomaly_ml" in df.columns else None
        if ml_clustered_df is not None:
            cluster_count = ml_clustered_df["cluster"].nunique()
        else:
            cluster_count = 0

        summary_markdown = f"""
- **Total log entries**: **{total_logs}**
- **INFO / WARN / ERROR / CRITICAL**: **{info_count} / {warn_count} / {error_count} / {critical_count}**
- **Rule-based anomalies (keywords)**: **{int(rule_anomalies)}**
- **ML anomalies (Isolation Forest)**: **{int(ml_anomalies)}**
- **KMeans clusters over ML anomalies**: **{cluster_count}**
- {top_pattern_text}
        """
        st.markdown(summary_markdown)

        st.subheader("⬇ Export Analyzed Report")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                "📄 Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "neurolog_report.csv",
                "text/csv"
            )
        
        with col2:
            json_data = export_to_json(df, include_metadata=True)
            st.download_button(
                "📋 Download JSON",
                json_data.encode("utf-8"),
                "neurolog_report.json",
                "application/json"
            )
        
        with col3:
            if PDF_AVAILABLE:
                pdf_data, pdf_error = export_to_pdf(df, title="NeuroLog Analysis Report")
                if pdf_error:
                    st.error(f"PDF Error: {pdf_error}")
                else:
                    st.download_button(
                        "📑 Download PDF",
                        pdf_data,
                        "neurolog_report.pdf",
                        "application/pdf"
                    )
            else:
                st.download_button(
                    "📑 Download PDF (Unavailable)",
                    b"",
                    "neurolog_report.pdf",
                    "application/pdf",
                    disabled=True,
                    help="Install reportlab library to enable PDF export"
                )
        
        with col4:
            st.download_button(
                "📝 Download Markdown",
                summary_markdown.encode("utf-8"),
                "neurolog_summary.md",
                "text/markdown",
            )

    with evaluation_tab:
        st.subheader("📏 Model Evaluation (Rule vs ML)")
        st.caption("Uses a small labelled dataset stored with the app to compare rule-based vs ML anomaly detection.")

        try:
            labelled_df = pd.read_csv("labelled_logs_example.csv")
        except FileNotFoundError:
            st.info("Labelled dataset 'labelled_logs_example.csv' not found. Add it to the app folder to enable evaluation.")
        else:
            st.dataframe(labelled_df, width='stretch', height=200)

            eval_keywords = rule_keywords_text
            eval_contamination = ml_contamination

            eval_df = labelled_df.copy()
            eval_df = detect_anomalies_rule_based(eval_df, eval_keywords)
            ml_eval_df = ml_detect_anomalies(eval_df.copy(), contamination=eval_contamination)
            eval_df["anomaly_ml"] = ml_eval_df["anomaly"].astype(str).str.upper().eq("YES")

            if "label" in eval_df.columns:
                true = eval_df["label"].astype(bool)

                def compute_metrics(pred):
                    tp = ((pred) & (true)).sum()
                    fp = ((pred) & (~true)).sum()
                    fn = ((~pred) & (true)).sum()
                    precision = tp / (tp + fp) if (tp + fp) else 0.0
                    recall = tp / (tp + fn) if (tp + fn) else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                    return precision, recall, f1

                rule_p, rule_r, rule_f1 = compute_metrics(eval_df["anomaly_rule"])
                ml_p, ml_r, ml_f1 = compute_metrics(eval_df["anomaly_ml"])

                metrics_df = pd.DataFrame(
                    [
                        ["Rule-based", rule_p, rule_r, rule_f1],
                        ["ML (Isolation Forest)", ml_p, ml_r, ml_f1],
                    ],
                    columns=["Method", "Precision", "Recall", "F1-score"],
                )

                st.subheader("Comparison of Detection Quality")
                st.dataframe(metrics_df.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-score": "{:.2f}"}), width='stretch')
            else:
                st.info("Labelled dataset must contain a 'label' column with ground truth anomalies (True/False).")

        st.subheader("🖊 Manual Labelling (current session)")
        if "manual_labels" not in st.session_state:
            st.session_state["manual_labels"] = {}

        if "anomaly_ml" in df.columns:
            current_anomalies = df[df["anomaly_ml"]].reset_index()
            if not current_anomalies.empty:
                selected_rows = st.multiselect(
                    "Mark the following ML anomalies as TRUE anomalies (for quick session-level evaluation):",
                    options=current_anomalies["index"].tolist(),
                    format_func=lambda idx: f"{idx}: {df.loc[idx, 'message'][:80]}",
                )
                st.session_state["manual_labels"] = {idx: True for idx in selected_rows}

                if selected_rows:
                    eval_indices = list(st.session_state["manual_labels"].keys())
                    eval_df_live = df.loc[eval_indices].copy()
                    true_live = eval_df_live.index.isin(eval_indices)

                    def compute_metrics_live(pred):
                        tp = ((pred) & (true_live)).sum()
                        fp = ((pred) & (~true_live)).sum()
                        fn = ((~pred) & (true_live)).sum()
                        precision = tp / (tp + fp) if (tp + fp) else 0.0
                        recall = tp / (tp + fn) if (tp + fn) else 0.0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                        return precision, recall, f1

                    rule_p2, rule_r2, rule_f12 = compute_metrics_live(eval_df_live["anomaly_rule"])
                    ml_p2, ml_r2, ml_f12 = compute_metrics_live(eval_df_live["anomaly_ml"])

                    live_metrics_df = pd.DataFrame(
                        [
                            ["Rule-based (session labels)", rule_p2, rule_r2, rule_f12],
                            ["ML (session labels)", ml_p2, ml_r2, ml_f12],
                        ],
                        columns=["Method", "Precision", "Recall", "F1-score"],
                    )

                    st.subheader("Session-level Metrics from Manual Labels")
                    st.dataframe(
                        live_metrics_df.style.format(
                            {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-score": "{:.2f}"}
                        ),
                        width='stretch',
                    )
            else:
                st.info("No ML anomalies available for manual labelling in this run.")

    with live_tab:
        st.subheader("📡 Enhanced Real-time Log Streaming")
        st.caption("Live log monitoring with real-time analysis")
        
        # Simplified streaming without threading
        col_stream1, col_stream2, col_stream3 = st.columns([2, 2, 3])
        
        with col_stream1:
            st.markdown("**Select log file**")
            available_logs = sorted([f for f in os.listdir(PROJECT_DIR) if f.lower().endswith(".log")])
            selected_log = st.selectbox(
                "Select log file",
                options=["(choose one)"] + available_logs,
                key="stream_log_select"
            )
        
        with col_stream2:
            st.markdown("**Manual path**")
            log_path_manual = st.text_input("Absolute path to log file", key="stream_log_path")
        
        with col_stream3:
            st.markdown("**Display options**")
            max_lines = st.number_input("Lines to display", min_value=10, max_value=500, value=50)
        
        if selected_log != "(choose one)":
            log_path = os.path.join(PROJECT_DIR, selected_log)
        else:
            log_path = log_path_manual.strip()
        
        if log_path and os.path.exists(log_path):
            st.subheader("🔴 Log File Content")
            
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                
                # Show last N lines
                recent_lines = lines[-max_lines:]
                log_content = "".join(recent_lines)
                
                # Parse for quick analysis
                try:
                    recent_df = parse_logs(log_content, use_batch=False)
                    if not recent_df.empty:
                        recent_df = detect_anomalies_rule_based(recent_df, "error, exception, failed, timeout")
                        
                        # Display with highlighting
                        for line in recent_lines[-20:]:  # Show last 20 lines
                            line = line.strip()
                            if any(word in line.lower() for word in ["error", "exception", "failed", "timeout"]):
                                st.markdown(f'<div style="background-color: rgba(239, 68, 68, 0.1); padding: 4px; border-radius: 4px; margin: 2px 0;">{line}</div>', unsafe_allow_html=True)
                            else:
                                st.text(line)
                        
                        # Quick stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            error_count = sum(1 for line in recent_lines if "error" in line.lower())
                            st.metric("🔴 Errors", error_count)
                        with col2:
                            warning_count = sum(1 for line in recent_lines if "warn" in line.lower())
                            st.metric("🟡 Warnings", warning_count)
                        with col3:
                            critical_count = sum(1 for line in recent_lines if "critical" in line.lower())
                            st.metric("⚠️ Critical", critical_count)
                    else:
                        st.code(log_content, language="text")
                except:
                    st.code(log_content, language="text")
                
            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            st.info("Select a log file to begin monitoring")

    with forecast_tab:
        st.subheader("🔮 Error Trend Forecasting")
        st.caption("Predict future error patterns using machine learning")
        
        if not ML_AVAILABLE:
            st.warning("⚠️ Machine learning libraries not available. Install scikit-learn to enable forecasting.")
        else:
            forecast_periods = st.slider("Forecast Period (hours)", min_value=6, max_value=48, value=24, step=6)
            
            with st.spinner("Generating forecast..."):
                forecast_df, error = forecast_error_trends(df, periods=forecast_periods)
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.success("✅ Forecast generated successfully")
                
                # Create forecast visualization
                fig = go.Figure()
                
                # Add historical data
                df_time = df.copy()
                df_time["timestamp"] = pd.to_datetime(df_time["timestamp"], errors="coerce")
                df_time = df_time.dropna(subset=["timestamp"])
                error_df = df_time[df_time["level"].isin(["ERROR", "CRITICAL"])]
                
                if not error_df.empty:
                    error_df["hour"] = error_df["timestamp"].dt.floor("h")
                    historical = error_df.groupby("hour").size().reset_index(name="count")
                    
                    fig.add_trace(go.Scatter(
                        x=historical["hour"],
                        y=historical["count"],
                        mode='lines+markers',
                        name='Historical Errors',
                        line=dict(color='red', width=2)
                    ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df["timestamp"],
                    y=forecast_df["forecast_count"],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="Error Count Forecast",
                    xaxis_title="Time",
                    yaxis_title="Error Count",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Forecast insights
                avg_forecast = forecast_df["forecast_count"].mean()
                max_forecast = forecast_df["forecast_count"].max()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Forecast", f"{avg_forecast:.1f}")
                with col2:
                    st.metric("Peak Forecast", f"{max_forecast:.1f}")
                with col3:
                    trend = "Increasing" if forecast_df["forecast_count"].iloc[-1] > forecast_df["forecast_count"].iloc[0] else "Decreasing"
                    st.metric("Trend", trend)

    with severity_tab:
        st.subheader("⚡ Anomaly Severity Analysis")
        st.caption("Advanced severity scoring and prioritization")
        
        if "anomaly_severity" not in df.columns:
            st.info("No severity scores calculated. Using default anomaly detection.")
        else:
            # Severity distribution
            severity_dist = df["anomaly_severity"].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Severity Distribution")
                fig = px.bar(
                    x=severity_dist.index,
                    y=severity_dist.values,
                    labels={"x": "Severity Score", "y": "Count"},
                    title="Anomaly Severity Distribution"
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.subheader("Severity Metrics")
                high_severity = (df["anomaly_severity"] >= 7).sum()
                medium_severity = ((df["anomaly_severity"] >= 4) & (df["anomaly_severity"] < 7)).sum()
                low_severity = (df["anomaly_severity"] < 4).sum()
                
                st.metric("High Severity (7-10)", high_severity, delta="🔴 Critical")
                st.metric("Medium Severity (4-6)", medium_severity, delta="🟡 Important")
                st.metric("Low Severity (1-3)", low_severity, delta="🟢 Minor")
            
            # Filter by severity
            severity_threshold = st.slider("Minimum Severity to Display", min_value=1, max_value=10, value=5)
            high_severity_anomalies = df[df["anomaly_severity"] >= severity_threshold]
            
            st.subheader(f"High Severity Anomalies (≥{severity_threshold})")
            if not high_severity_anomalies.empty:
                display_cols = ["timestamp", "level", "message", "anomaly_severity"]
                if "anomaly_ml" in high_severity_anomalies.columns:
                    display_cols.append("anomaly_ml")
                if "anomaly_rule" in high_severity_anomalies.columns:
                    display_cols.append("anomaly_rule")
                
                st.dataframe(
                    high_severity_anomalies[display_cols].sort_values("anomaly_severity", ascending=False),
                    width='stretch'
                )
            else:
                st.info("No high severity anomalies found with current threshold.")
else:
    if page not in ("home", "login", "register", "history"):
        st.info(
            "📂 To get started, use the **Configuration** sidebar on the left to upload a log file and choose the correct log format."
        )
