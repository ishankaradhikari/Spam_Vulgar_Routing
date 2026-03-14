"""
config.py - Application Configuration
Centralised config so switching to MySQL/PostgreSQL only requires
changing DATABASE_URI and installing the appropriate driver.
"""

import os
import secrets
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # ── Security ──────────────────────────────────────────────────────────────
  # Original default secret key (replace in production!)
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production-please")

    # Session cookie security
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"  # Use 'Strict' for stricter CSRF protection
    SESSION_COOKIE_SECURE = False      # False for local dev; True in production with HTTPS

    # ── Database ───────────────────────────────────────────────────────────────
    # SQLite (default).  Swap for:
    #   mysql+pymysql://user:pass@host/db
    #   postgresql://user:pass@host/db
    DATABASE_URI = os.environ.get(
        "DATABASE_URI",
        f"sqlite:///{os.path.join(BASE_DIR, 'database.db')}"
    )

    # ── Uploads ────────────────────────────────────────────────────────────────
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024   # 5 MB
    ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "gif", "txt", "doc", "docx"}

    # ── Pagination ─────────────────────────────────────────────────────────────
    MESSAGES_PER_PAGE = 20

