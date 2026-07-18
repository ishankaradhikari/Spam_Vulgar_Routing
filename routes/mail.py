"""
routes/mail.py - Core messaging blueprint
Handles: inbox, sent, spam, trash, compose, view, delete, star, mark-read,
         reply, search, pagination, and file attachments.
"""

import os
from functools import wraps
from flask import (
    Blueprint, render_template, request, redirect,
    url_for, session, flash, current_app, jsonify, abort
)
from database import get_db
# NEW: import classify_message (full pipeline) AND the backward-compat is_spam shim
from spam_filter import classify_message, is_spam

mail_bp = Blueprint("mail", __name__)


# ─────────────────────────────────────────────────────────────────────────────
# Auth guard decorator
# ─────────────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "info")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _folder_counts(user_id):
    """Return unread counts per folder for the sidebar badge."""
    db = get_db()
    rows = db.execute(
        """SELECT folder, COUNT(*) as cnt
           FROM messages
           WHERE recipient_id=? AND is_read=0 AND deleted_at IS NULL
             AND folder IN ('inbox','spam')
           GROUP BY folder""",
        (user_id,)
    ).fetchall()
    counts = {"inbox": 0, "sent": 0, "spam": 0, "trash": 0}
    for r in rows:
        counts[r["folder"]] = r["cnt"]
    return counts




def _get_message_or_404(msg_id, user_id):
    """Fetch a message that belongs to the current user (sender or recipient)."""
    db  = get_db()
    msg = db.execute(
        """SELECT m.*, 
                  s.username  AS sender_username,
                  s.display_name AS sender_display,
                  s.avatar_color AS sender_color,
                  r.username  AS recipient_username,
                  r.display_name AS recipient_display
           FROM messages m
           JOIN users s ON s.id = m.sender_id
           JOIN users r ON r.id = m.recipient_id
           WHERE m.id=? AND (m.sender_id=? OR m.recipient_id=?)
             AND (m.deleted_at IS NULL OR m.folder='trash')""",
        (msg_id, user_id, user_id)
    ).fetchone()
    if not msg:
        abort(404)
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Folder views
# ─────────────────────────────────────────────────────────────────────────────

def _render_folder(folder_name, user_id, template="mailbox.html"):
    db      = get_db()
    page    = request.args.get("page", 1, type=int)
    q       = request.args.get("q", "").strip()
    per_page = current_app.config["MESSAGES_PER_PAGE"]
    offset  = (page - 1) * per_page

    # Determine which ownership clause to filter by.
    if folder_name == "sent":
        owner_clause = "m.sender_id=?"
        params = [user_id, folder_name]
    elif folder_name == "trash":
        owner_clause = "(m.sender_id=? OR m.recipient_id=?)"
        params = [user_id, user_id, folder_name]
    else:
        owner_clause = "m.recipient_id=?"
        params = [user_id, folder_name]

    deleted_clause = "AND (m.deleted_at IS NULL OR m.folder='trash')" if folder_name == "trash" else "AND m.deleted_at IS NULL"

    search_clause = ""
    if q:
        search_clause = " AND (m.subject LIKE ? OR m.body LIKE ?)"
        params += [f"%{q}%", f"%{q}%"]

    params_count = params.copy()
    params += [per_page, offset]

    base_query = f"""
        FROM messages m
        JOIN users s ON s.id = m.sender_id
        JOIN users r ON r.id = m.recipient_id
        WHERE {owner_clause} AND m.folder=? {deleted_clause}
        {search_clause}
    """

    total = db.execute(f"SELECT COUNT(*) {base_query}", params_count).fetchone()[0]

    messages = db.execute(
        f"""SELECT m.*,
                   s.username  AS sender_username,
                   s.display_name AS sender_display,
                   s.avatar_color AS sender_color,
                   r.username  AS recipient_username,
                   r.display_name AS recipient_display
            {base_query}
            ORDER BY m.sent_at DESC
            LIMIT ? OFFSET ?""",
        params
    ).fetchall()

    total_pages = max(1, (total + per_page - 1) // per_page)

    return render_template(
        template,
        messages=messages,
        folder=folder_name,
        folder_counts=_folder_counts(user_id),
        page=page,
        total_pages=total_pages,
        q=q,
        total=total,
    )


@mail_bp.route("/")
def index():
    if "user_id" not in session:
        return render_template("landing.html")
    return redirect(url_for("mail.inbox"))


@mail_bp.route("/inbox")
@login_required
def inbox():
    return _render_folder("inbox", session["user_id"])


@mail_bp.route("/sent")
@login_required
def sent():
    return _render_folder("sent", session["user_id"])


@mail_bp.route("/spam")
@login_required
def spam():
    return _render_folder("spam", session["user_id"])


@mail_bp.route("/trash")
@login_required
def trash():
    return _render_folder("trash", session["user_id"])


# ─────────────────────────────────────────────────────────────────────────────
# Compose / Send
# ─────────────────────────────────────────────────────────────────────────────

@mail_bp.route("/compose", methods=["GET", "POST"])
@login_required
def compose():
    db = get_db()

    # Prefill reply fields
    prefill = {
        "to":      request.args.get("to", ""),
        "subject": request.args.get("subject", ""),
        "body":    request.args.get("body", ""),
        "thread_id": request.args.get("thread_id", ""),
    }

    if request.method == "POST":
        to_username = request.form.get("to", "").strip()
        subject     = request.form.get("subject", "").strip() or "(no subject)"
        body        = request.form.get("body", "").strip()
        thread_id   = request.form.get("thread_id") or None

        # Validate recipient
        recipient = db.execute(
            "SELECT * FROM users WHERE username=?", (to_username,)
        ).fetchone()
        if not recipient:
            flash(f'User "{to_username}" not found.', "error")
            return render_template("compose.html",
                                   folder_counts=_folder_counts(session["user_id"]),
                                   prefill={**prefill, "to": to_username,
                                            "subject": subject, "body": body})

        # ── Full ML Moderation Pipeline (Parts 1–6) ───────────────────────────
        # classify_message() runs:
        #   1. Existing spam NB → probability + threshold decision (Part 1)
        #   2. New vulgar classifier → probability + threshold decision (Part 4)
        #   3. Priority decision: VULGAR > SPAM > CLEAN (Part 5)
        ml = classify_message(subject, body)

        # Part 6: Act on decision
        if ml['blocked']:
            # ── VULGAR: block delivery, store in moderation_log ────────────────
            db.execute(
                """INSERT INTO moderation_log
                   (sender_id, recipient_id, subject, body,
                    vulgar_probability, spam_probability, moderation_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session["user_id"], recipient["id"], subject, body,
                 ml['vulgar_probability'], ml['spam_probability'],
                 ml['decision_reason'])
            )
            db.commit()
            flash(f'⚠️ {ml["warning"]}', "error")
            return render_template("compose.html",
                                   folder_counts=_folder_counts(session["user_id"]),
                                   prefill={**prefill, "to": to_username,
                                            "subject": subject, "body": body},
                                   users=db.execute(
                                       "SELECT username, display_name FROM users WHERE id != ? ORDER BY username",
                                       (session["user_id"],)).fetchall())

        # SPAM or CLEAN: store normally with ML metadata
        inbox_folder = ml['folder']
        sender_id    = session["user_id"]
        recipient_id = recipient["id"]

        # Insert sender copy first (sent folder, always readable)
        sender_cur = db.execute(
            """INSERT INTO messages
               (thread_id, sender_id, recipient_id, subject, body, folder,
                is_read, attachment,
                spam_flag, spam_probability, ham_probability, spam_confidence,
                vulgar_probability, is_vulgar, is_blocked, moderation_reason)
               VALUES (?, ?, ?, ?, ?, 'sent', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (thread_id, sender_id, recipient_id,
             subject, body, None,
             ml['spam_flag'],
             ml['spam_probability'],
             ml['ham_probability'],
             ml['confidence'],
             ml['vulgar_probability'],
             int(ml['is_vulgar']),
             int(ml['blocked']),
             ml['decision_reason'])
        )
        sender_msg_id = sender_cur.lastrowid

        # Insert recipient copy with full ML metadata columns.
        # This preserves the inbox/spam/moderation routing for recipients,
        # even when sending a message to yourself.
        cur = db.execute(
            """INSERT INTO messages
               (thread_id, sender_id, recipient_id, subject, body, folder,
                attachment,
                spam_flag, spam_probability, ham_probability, spam_confidence,
                vulgar_probability, is_vulgar, is_blocked, moderation_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (thread_id or sender_msg_id, sender_id, recipient_id, subject, body, inbox_folder,
             None,
             ml['spam_flag'],
             ml['spam_probability'],
             ml['ham_probability'],
             ml['confidence'],
             ml['vulgar_probability'],
             int(ml['is_vulgar']),
             int(ml['blocked']),
             ml['decision_reason'])
        )
        recipient_msg_id = cur.lastrowid

        # Log spam to audit table for the recipient copy
        if ml['is_spam']:
            db.execute(
                "INSERT INTO spam_log (message_id, reason) VALUES (?, ?)",
                (recipient_msg_id, ml['decision_reason'])
            )

        # If root message, set thread_id for all copies
        if not thread_id:
            db.execute("UPDATE messages SET thread_id=? WHERE thread_id IS NULL AND (sender_id=? OR recipient_id=?)",
                       (sender_msg_id, sender_id, sender_id))

        db.commit()

        if ml['is_spam']:
            flash(f'Message sent (auto-filtered to spam: {ml["decision_reason"]}).', "warning")
        else:
            flash("Message sent!", "success")
        return redirect(url_for("mail.sent"))

    # GET - load user list for autocomplete
    users = db.execute(
        "SELECT username, display_name FROM users WHERE id != ? ORDER BY username",
        (session["user_id"],)
    ).fetchall()

    return render_template("compose.html",
                           folder_counts=_folder_counts(session["user_id"]),
                           prefill=prefill,
                           users=users)


# ─────────────────────────────────────────────────────────────────────────────
# View message
# ─────────────────────────────────────────────────────────────────────────────

@mail_bp.route("/message/<int:msg_id>")
@login_required
def view_message(msg_id):
    user_id = session["user_id"]
    db      = get_db()
    msg     = _get_message_or_404(msg_id, user_id)

    # Mark as read if recipient
    if msg["recipient_id"] == user_id and not msg["is_read"]:
        db.execute("UPDATE messages SET is_read=1 WHERE id=?", (msg_id,))
        db.commit()

    # Thread messages - one row per logical message (deduplicated by sender+time).
    # Each send creates two DB rows (inbox copy + sent copy); we show only the
    # inbox/recipient copy when it exists, falling back to the sent copy.
    deleted_clause = "AND (m.deleted_at IS NULL OR m.folder='trash')" if msg["folder"] == "trash" else "AND m.deleted_at IS NULL"
    thread_raw = db.execute(
        f"""SELECT m.*, 
                  s.username  AS sender_username,
                  s.display_name AS sender_display,
                  s.avatar_color AS sender_color
           FROM messages m
           JOIN users s ON s.id = m.sender_id
           WHERE m.thread_id=? {deleted_clause}
             AND (m.sender_id=? OR m.recipient_id=?)
           ORDER BY m.sent_at ASC, m.folder ASC""",
        (msg["thread_id"] or msg_id, user_id, user_id)
    ).fetchall()

    # Deduplicate: keep the inbox/spam copy over the sent copy for each
    # (sender_id, sent_at) pair so each logical message appears only once.
    seen = {}
    for t in thread_raw:
        key = (t["sender_id"], t["sent_at"])
        if key not in seen or t["folder"] != "sent":
            seen[key] = t
    thread = list(seen.values())

    return render_template("view_message.html",
                           msg=msg,
                           thread=thread,
                           folder_counts=_folder_counts(user_id))


# ─────────────────────────────────────────────────────────────────────────────
# Actions: delete, restore, mark-read, star, move-to-spam
# ─────────────────────────────────────────────────────────────────────────────

@mail_bp.route("/message/<int:msg_id>/delete", methods=["POST"])
@login_required
def delete_message(msg_id):
    user_id = session["user_id"]
    db      = get_db()
    msg     = _get_message_or_404(msg_id, user_id)

    if msg["folder"] == "trash":
        # Permanent delete
        if msg["sender_id"] == user_id and msg["recipient_id"] == user_id:
            db.execute(
                """DELETE FROM messages
                   WHERE folder='trash'
                     AND sender_id=?
                     AND recipient_id=?
                     AND subject=?
                     AND body=?
                     AND sent_at=?""",
                (user_id, user_id, msg["subject"], msg["body"], msg["sent_at"])
            )
        else:
            db.execute("DELETE FROM messages WHERE id=?", (msg_id,))
        flash("Message permanently deleted.", "info")
    else:
        # Move to trash (soft delete)
        db.execute(
            "UPDATE messages SET folder='trash', deleted_at=CURRENT_TIMESTAMP WHERE id=?",
            (msg_id,)
        )
        flash("Message moved to Trash.", "info")

    db.commit()
    return redirect(request.referrer or url_for("mail.inbox"))


@mail_bp.route("/message/<int:msg_id>/restore", methods=["POST"])
@login_required
def restore_message(msg_id):
    user_id = session["user_id"]
    db      = get_db()
    msg     = db.execute(
        "SELECT * FROM messages WHERE id=? AND (sender_id=? OR recipient_id=?)",
        (msg_id, user_id, user_id)
    ).fetchone()
    if not msg:
        abort(404)

    restore_to = "sent" if msg["sender_id"] == user_id else "inbox"
    db.execute(
        "UPDATE messages SET folder=?, deleted_at=NULL WHERE id=?",
        (restore_to, msg_id)
    )
    db.commit()
    flash("Message restored.", "success")
    return redirect(url_for("mail.trash"))


@mail_bp.route("/message/<int:msg_id>/toggle-read", methods=["POST"])
@login_required
def toggle_read(msg_id):
    user_id = session["user_id"]
    db      = get_db()
    msg     = _get_message_or_404(msg_id, user_id)
    db.execute("UPDATE messages SET is_read=? WHERE id=?",
               (0 if msg["is_read"] else 1, msg_id))
    db.commit()
    return redirect(request.referrer or url_for("mail.inbox"))


@mail_bp.route("/message/<int:msg_id>/toggle-star", methods=["POST"])
@login_required
def toggle_star(msg_id):
    user_id = session["user_id"]
    db      = get_db()
    msg     = _get_message_or_404(msg_id, user_id)
    db.execute("UPDATE messages SET is_starred=? WHERE id=?",
               (0 if msg["is_starred"] else 1, msg_id))
    db.commit()
    return redirect(request.referrer or url_for("mail.inbox"))


@mail_bp.route("/message/<int:msg_id>/mark-spam", methods=["POST"])
@login_required
def mark_spam(msg_id):
    user_id = session["user_id"]
    db      = get_db()
    _get_message_or_404(msg_id, user_id)   # ownership check
    db.execute("UPDATE messages SET folder='spam' WHERE id=?", (msg_id,))
    db.commit()
    flash("Message marked as spam.", "warning")
    return redirect(request.referrer or url_for("mail.inbox"))


@mail_bp.route("/message/<int:msg_id>/not-spam", methods=["POST"])
@login_required
def not_spam(msg_id):
    user_id = session["user_id"]
    db      = get_db()
    _get_message_or_404(msg_id, user_id)
    db.execute("UPDATE messages SET folder='inbox' WHERE id=?", (msg_id,))
    db.commit()
    flash("Message moved to Inbox.", "success")
    return redirect(request.referrer or url_for("mail.spam"))


# ─────────────────────────────────────────────────────────────────────────────
# Bulk actions (AJAX-friendly)
# ─────────────────────────────────────────────────────────────────────────────

@mail_bp.route("/bulk-action", methods=["POST"])
@login_required
def bulk_action():
    user_id = session["user_id"]
    db      = get_db()
    ids     = request.form.getlist("msg_ids")
    action  = request.form.get("action")
    current_folder = request.form.get("current_folder")

    if not ids or action not in ("delete", "read", "unread", "spam", "star"):
        flash("Invalid bulk action.", "error")
        return redirect(request.referrer or url_for("mail.inbox"))

    placeholders = ",".join("?" for _ in ids)
    ownership    = f"""
        id IN ({placeholders})
        AND (sender_id=? OR recipient_id=?)
    """
    all_params = ids + [user_id, user_id]

    if action == "delete":
        if current_folder == "trash":
            db.execute(f"DELETE FROM messages WHERE {ownership}", all_params)
            flash(f"{len(ids)} message(s) permanently deleted.", "info")
        else:
            db.execute(
                f"UPDATE messages SET folder='trash', deleted_at=CURRENT_TIMESTAMP WHERE {ownership}",
                all_params
            )
            flash(f"{len(ids)} message(s) moved to Trash.", "info")
    elif action == "read":
        db.execute(f"UPDATE messages SET is_read=1 WHERE {ownership}", all_params)
        flash(f"{len(ids)} marked as read.", "info")
    elif action == "unread":
        db.execute(f"UPDATE messages SET is_read=0 WHERE {ownership}", all_params)
        flash(f"{len(ids)} marked as unread.", "info")
    elif action == "spam":
        db.execute(f"UPDATE messages SET folder='spam' WHERE {ownership}", all_params)
        flash(f"{len(ids)} moved to Spam.", "warning")
    elif action == "star":
        db.execute(f"UPDATE messages SET is_starred=1 WHERE {ownership}", all_params)
        flash(f"{len(ids)} starred.", "info")

    db.commit()
    return redirect(request.referrer or url_for("mail.inbox"))


# ─────────────────────────────────────────────────────────────────────────────
# Profile page
# ─────────────────────────────────────────────────────────────────────────────

@mail_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    db      = get_db()
    user_id = session["user_id"]
    user    = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()

    if request.method == "POST":
        display_name = request.form.get("display_name", "").strip() or user["username"]
        avatar_color = request.form.get("avatar_color", user["avatar_color"])
        db.execute(
            "UPDATE users SET display_name=?, avatar_color=? WHERE id=?",
            (display_name, avatar_color, user_id)
        )
        db.commit()
        session["display_name"] = display_name
        session["avatar_color"] = avatar_color
        flash("Profile updated.", "success")
        return redirect(url_for("mail.profile"))

    # Stats
    stats = db.execute(
        """SELECT
             SUM(CASE WHEN folder='inbox' AND recipient_id=? THEN 1 ELSE 0 END) AS inbox_count,
             SUM(CASE WHEN folder='sent'  AND sender_id=?    THEN 1 ELSE 0 END) AS sent_count,
             SUM(CASE WHEN folder='spam'  AND recipient_id=? THEN 1 ELSE 0 END) AS spam_count
           FROM messages WHERE deleted_at IS NULL""",
        (user_id, user_id, user_id)
    ).fetchone()

    return render_template("profile.html",
                           user=user, stats=stats,
                           folder_counts=_folder_counts(user_id))


# ─────────────────────────────────────────────────────────────────────────────
# REST API endpoints
# ─────────────────────────────────────────────────────────────────────────────

@mail_bp.route("/api/users/search")
@login_required
def api_user_search():
    """Autocomplete: returns JSON list of usernames matching ?q="""
    q   = request.args.get("q", "").strip()
    db  = get_db()
    if len(q) < 1:
        return jsonify([])
    rows = db.execute(
        "SELECT username, display_name FROM users WHERE username LIKE ? AND id != ? LIMIT 10",
        (f"{q}%", session["user_id"])
    ).fetchall()
    return jsonify([{"username": r["username"], "display_name": r["display_name"]} for r in rows])


@mail_bp.route("/api/folder-counts")
@login_required
def api_folder_counts():
    return jsonify(_folder_counts(session["user_id"]))
