# PulseMailer Enhanced

A Flask-based mail application with real-time messaging, ML statistics, and a redesigned light UI.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (MUST use app.py directly, NOT flask run)
python app.py
```

Open: http://127.0.0.1:5000

---

## What Was Added

### Part 1 — Real-Time Communication (Flask-SocketIO)

| Feature | How it works |
|---|---|
| Live message delivery | After sending, server pushes `new_message` event to recipient's browser room |
| Live inbox update | Recipient's inbox prepends the new row without refresh |
| Typing indicator | Compose textarea emits `typing` event; recipient sees "X is typing…" |
| Online/offline status | Connected users tracked in `_online_users` dict; status dots in autocomplete |
| Notifications | Bell icon shows badge + dropdown with recent message previews |

**Socket events used:**

- `connect` / `disconnect` — track online users
- `typing` — forward typing status to recipient
- `new_message_sent` — client → server after compose
- `new_message` — server → recipient browser
- `online_users` — broadcast updated online list

**Key files changed:**
- `app.py` — SocketIO init + event handlers
- `routes/mail.py` — added `/api/send` JSON endpoint
- `static/js/main.js` — `initSocket()`, `initTypingEmit()`, etc.
- `templates/base.html` — notification bell, typing bar, SocketIO `<script>`

### Part 2 — ML Statistics Dashboard (`/model_stats`)

Evaluates the Naive Bayes classifier against the full training dataset.

**Metrics computed:**

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1 Score  = 2 × Precision × Recall / (Precision + Recall)
```

Where:
- **TP** — model predicted SPAM, label is spam
- **TN** — model predicted HAM, label is ham
- **FP** — model predicted SPAM, label is ham (false alarm)
- **FN** — model predicted HAM, label is spam (missed spam)

Threshold: `spam_probability >= 0.70` (same as `spam_filter.py`).

> **Note:** Train-set evaluation inflates metrics. For a true assessment, split the dataset into train/test sets before training.

**New files:**
- `routes/stats.py` — `compute_stats()` + `/model_stats` route
- `templates/model_stats.html` — dashboard with confusion matrix, metric cards, sample table

### Part 3 — UI Redesign (Light Theme)

- Background: `#f4f6fb` (light gray-blue)
- Surface: `#ffffff` (white cards)
- Accent: `#528fff` (soft blue) + `#2ecba1` (teal)
- Font: Inter (clean, modern sans-serif)
- Sidebar: white with subtle border, rounded nav items
- Message rows: white cards with hover state
- Buttons: flat with clear visual hierarchy
- Auth pages: centered card layout
- All existing dark-theme CSS replaced in `static/css/style.css`

### Part 4 — Existing Features Preserved

All original functionality is intact:
- User authentication (register/login/logout)
- Database schema and migrations
- Naive Bayes spam detection (unchanged)
- Vulgar word classifier (unchanged)
- `spam_filter.py` pipeline (unchanged)
- Message threading, starring, bulk actions
- File attachments
- Autocomplete

---

## File Changes Summary

| File | Status | What changed |
|---|---|---|
| `app.py` | Modified | SocketIO init, socket event handlers, stats blueprint |
| `routes/mail.py` | Modified | `_do_send_message()` helper, `/api/send`, `/api/online-users` |
| `routes/stats.py` | **New** | ML stats blueprint and `compute_stats()` |
| `templates/base.html` | Modified | Notification bell, typing bar, SocketIO script tag |
| `templates/model_stats.html` | **New** | ML dashboard page |
| `templates/login.html` | Modified | Light theme |
| `templates/register.html` | Modified | Light theme |
| `templates/compose.html` | Modified | Socket emit after send |
| `templates/profile.html` | Modified | Light theme |
| `static/css/style.css` | Modified | Full light theme redesign |
| `static/js/main.js` | Modified | `initSocket()`, notifications, typing, online status |
| `requirements.txt` | Modified | Added `flask-socketio==5.3.6` |

---

## Running Notes

- Use `python app.py` (not `flask run`) — SocketIO requires `socketio.run()`
- The SQLite database file (`database.db`) is preserved as-is
- No database schema changes were needed for this enhancement
