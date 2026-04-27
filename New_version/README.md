# PulseMailer — Full-Stack Messaging Web Application

A production-ready, mini email system built with Flask + SQLite.

## Features

- ✅ User registration, login, logout with password hashing (werkzeug)
- ✅ Compose and send messages between registered users
- ✅ Inbox, Sent, Spam, Trash folders
- ✅ Automatic spam detection (keyword scoring system)
- ✅ Message threading (replies linked together)
- ✅ Mark read/unread, star messages
- ✅ File attachments (PDF, images, documents, up to 5 MB)
- ✅ Bulk actions (mark read, delete, spam, star)
- ✅ Full-text search across messages
- ✅ Pagination (20 messages per page)
- ✅ User profile page with avatar colour picker
- ✅ REST API endpoints (`/api/users/search`, `/api/folder-counts`)
- ✅ Responsive design (mobile sidebar)
- ✅ SQL injection protection (parameterised queries throughout)
- ✅ Session security (HttpOnly, SameSite)

---

## Project Structure

```
messaging_app/
├── app.py                # Flask app factory + entry point
├── config.py             # Centralised configuration
├── database.py           # DB helpers, schema, init
├── spam_filter.py        # Keyword-based spam scoring
├── requirements.txt      # Python dependencies
├── database.db           # SQLite file (auto-created on first run)
│
├── routes/
│   ├── __init__.py
│   ├── auth.py           # /register  /login  /logout
│   └── mail.py           # All mailbox routes + REST API
│
├── templates/
│   ├── base.html         # App shell with sidebar
│   ├── login.html
│   ├── register.html
│   ├── mailbox.html      # Inbox / Sent / Spam / Trash list
│   ├── view_message.html # Message detail + thread + reply
│   ├── compose.html
│   └── profile.html
│
└── static/
    ├── css/style.css     # Full dark-theme UI
    ├── js/main.js        # Autocomplete, select-all, etc.
    └── uploads/          # Attachment storage (auto-created)
```

---

## Quickstart

### 1. Prerequisites

- Python 3.9 or higher
- pip

### 2. Set up a virtual environment (recommended)

```bash
cd messaging_app
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

The database (`database.db`) and the `static/uploads/` directory are created automatically on first launch.

Open your browser at: **http://localhost:5000**

---

## Environment Variables (optional)

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `change-me-in-production-please` | Flask session signing key — **change this in production** |
| `DATABASE_URI` | `sqlite:///database.db` | Database URL |

Example for production:
```bash
export SECRET_KEY="your-super-secret-random-string"
python app.py
```

---

## Migrating to MySQL or PostgreSQL

1. Install the driver:
   - MySQL: `pip install PyMySQL`
   - PostgreSQL: `pip install psycopg2-binary`

2. Set the `DATABASE_URI` environment variable:
   ```bash
   # MySQL
   export DATABASE_URI="mysql+pymysql://user:password@localhost/pulsemail"

   # PostgreSQL
   export DATABASE_URI="postgresql://user:password@localhost/pulsemail"
   ```

3. The schema in `database.py` uses standard SQL — for MySQL/PostgreSQL swap
   `sqlite3` for SQLAlchemy (`pip install SQLAlchemy`) and update `get_db()` accordingly.
   The table definitions remain the same.

---

## Database Schema

```sql
CREATE TABLE users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    display_name  TEXT,
    avatar_color  TEXT    DEFAULT '#6c63ff',
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login    DATETIME
);

CREATE TABLE messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id     INTEGER,
    sender_id     INTEGER NOT NULL REFERENCES users(id),
    recipient_id  INTEGER NOT NULL REFERENCES users(id),
    subject       TEXT    NOT NULL,
    body          TEXT    NOT NULL,
    folder        TEXT    NOT NULL DEFAULT 'inbox',
    is_read       INTEGER NOT NULL DEFAULT 0,
    is_starred    INTEGER NOT NULL DEFAULT 0,
    attachment    TEXT,
    sent_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted_at    DATETIME
);

CREATE TABLE spam_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER REFERENCES messages(id),
    reason     TEXT,
    flagged_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/users/search?q=<query>` | Autocomplete username search (JSON) |
| `GET` | `/api/folder-counts` | Unread counts per folder (JSON) |

---

## Spam Detection

The spam scorer in `spam_filter.py` assigns points based on:
- Keyword hits (e.g. "free money", "act now", "win a prize") — **+3 pts each**
- Excessive caps (>60 % of text is uppercase) — **+2 pts**
- Exclamation marks — **+1 pt each (max 3)**
- Suspicious URLs — **+1 pt each (max 2)**
- Very short bait messages — **+1 pt**

Messages scoring **≥ 5** are auto-routed to the recipient's Spam folder.

Add or remove keywords by editing `SPAM_KEYWORDS` in `config.py`.

---

## Security Notes

- Passwords are hashed with `werkzeug.security.generate_password_hash` (PBKDF2-SHA256)
- All database queries use parameterised statements — no string interpolation
- Sessions are `HttpOnly` and `SameSite=Lax`
- File uploads are sanitised with `werkzeug.utils.secure_filename` + extension allowlist
- Change `SECRET_KEY` before any public deployment
