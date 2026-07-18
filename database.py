"""
database.py - Database Abstraction Layer
Uses sqlite3 directly for SQLite; swap get_db() for SQLAlchemy session
when migrating to MySQL / PostgreSQL.
"""

import sqlite3
import os
from flask import g, current_app


# ─────────────────────────────────────────────────────────────────────────────
# Connection helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db():
    """Return a per-request SQLite connection stored on Flask's g object."""
    if "db" not in g:
        db_path = current_app.config["DATABASE_URI"].replace("sqlite:///", "")
        g.db = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row          # access columns by name
        g.db.execute("PRAGMA foreign_keys = ON") # enforce FK constraints
    return g.db


def close_db(e=None):
    """Teardown: close the connection at end of request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Schema – create tables on first run
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = """
-- ── Users ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT    NOT NULL,
    display_name  TEXT,
    avatar_color  TEXT    DEFAULT '#6c63ff',
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login    DATETIME
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);

-- ── Messages ─────────────────────────────────────────────────────────────────
-- Each logical message creates TWO rows: one for sender (folder='sent'),
-- one for recipient (folder='inbox', 'spam', or 'moderation').
-- thread_id links replies together.
--
-- NEW COLUMNS (v2 - ML moderation enhancement):
--   spam_flag          : 1 if message was classified as spam, 0 otherwise
--   spam_probability   : P(spam|message) from Naive Bayes  [0.0 - 1.0]
--   ham_probability    : P(ham|message)                    [0.0 - 1.0]
--   spam_confidence    : |P(spam) - P(ham)|, model certainty [0.0 - 1.0]
--   vulgar_probability : P(vulgar|message)                 [0.0 - 1.0]
--   is_vulgar          : 1 if message was classified as vulgar, 0 otherwise
--   is_blocked         : 1 if message was blocked from delivery (vulgar)
--   moderation_reason  : human-readable explanation of the ML decision
CREATE TABLE IF NOT EXISTS messages (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id           INTEGER,
    sender_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recipient_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    subject             TEXT    NOT NULL DEFAULT '(no subject)',
    body                TEXT    NOT NULL DEFAULT '',
    folder              TEXT    NOT NULL DEFAULT 'inbox',
    is_read             INTEGER NOT NULL DEFAULT 0,
    is_starred          INTEGER NOT NULL DEFAULT 0,
    attachment          TEXT,
    sent_at             DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted_at          DATETIME,
    -- ── ML moderation columns (new) ──────────────────────────────────────────
    spam_flag           INTEGER NOT NULL DEFAULT 0,
    spam_probability    REAL    NOT NULL DEFAULT 0.0,
    ham_probability     REAL    NOT NULL DEFAULT 0.0,
    spam_confidence     REAL    NOT NULL DEFAULT 0.0,
    vulgar_probability  REAL    NOT NULL DEFAULT 0.0,
    is_vulgar           INTEGER NOT NULL DEFAULT 0,
    is_blocked          INTEGER NOT NULL DEFAULT 0,
    moderation_reason   TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_recipient ON messages(recipient_id, folder);
CREATE INDEX IF NOT EXISTS idx_messages_sender    ON messages(sender_id,    folder);
CREATE INDEX IF NOT EXISTS idx_messages_thread    ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_sent_at   ON messages(sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_spam_flag ON messages(spam_flag);
CREATE INDEX IF NOT EXISTS idx_messages_blocked   ON messages(is_blocked);

-- ── Spam log ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS spam_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    reason     TEXT,
    flagged_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── Moderation log (NEW) ──────────────────────────────────────────────────────
-- Stores all BLOCKED messages (vulgar) for admin review.
-- These messages are never delivered to the recipient's inbox/spam.
--
-- Columns:
--   id                : surrogate key
--   sender_id         : who sent the blocked message
--   recipient_id      : intended recipient
--   subject / body    : original message content
--   vulgar_probability: model confidence in vulgar classification
--   spam_probability  : model's spam assessment (informational)
--   raw_text          : subject + body combined (for auditing)
--   blocked_at        : timestamp of block
CREATE TABLE IF NOT EXISTS moderation_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recipient_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    subject             TEXT    NOT NULL DEFAULT '',
    body                TEXT    NOT NULL DEFAULT '',
    vulgar_probability  REAL    NOT NULL DEFAULT 0.0,
    spam_probability    REAL    NOT NULL DEFAULT 0.0,
    moderation_reason   TEXT,
    blocked_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_modlog_sender    ON moderation_log(sender_id);
CREATE INDEX IF NOT EXISTS idx_modlog_recipient ON moderation_log(recipient_id);
CREATE INDEX IF NOT EXISTS idx_modlog_blocked   ON moderation_log(blocked_at DESC);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Migration helper - adds new columns to existing databases
# ─────────────────────────────────────────────────────────────────────────────

_MIGRATION_COLUMNS = [
    # (table, column_name, column_definition)
    ("messages", "spam_flag",          "INTEGER NOT NULL DEFAULT 0"),
    ("messages", "spam_probability",   "REAL    NOT NULL DEFAULT 0.0"),
    ("messages", "ham_probability",    "REAL    NOT NULL DEFAULT 0.0"),
    ("messages", "spam_confidence",    "REAL    NOT NULL DEFAULT 0.0"),
    ("messages", "vulgar_probability", "REAL    NOT NULL DEFAULT 0.0"),
    ("messages", "is_vulgar",          "INTEGER NOT NULL DEFAULT 0"),
    ("messages", "is_blocked",         "INTEGER NOT NULL DEFAULT 0"),
    ("messages", "moderation_reason",  "TEXT"),
]


def migrate_db(app) -> None:
    """
    Add new ML columns to an existing database without data loss.
    SQLite supports ADD COLUMN but not DROP/MODIFY, so this is safe
    to run on an already-populated database.
    """
    db_path = app.config["DATABASE_URI"].replace("sqlite:///", "")
    conn    = sqlite3.connect(db_path)
    cursor  = conn.cursor()

    for table, col, col_def in _MIGRATION_COLUMNS:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
            print(f"[DB Migration] Added column {table}.{col}")
        except sqlite3.OperationalError:
            pass   # column already exists - safe to ignore

    conn.commit()
    conn.close()


def init_db(app):
    """
    Create all tables and register teardown hook.
    Also runs migrate_db() to add new ML columns to any existing database.
    """
    db_path = app.config["DATABASE_URI"].replace("sqlite:///", "")
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()

    # Add new ML columns to pre-existing database (safe / idempotent)
    migrate_db(app)

    app.teardown_appcontext(close_db)
