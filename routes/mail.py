"""
routes/mail.py - Core messaging blueprint (ENHANCED)
=====================================================
SOCKET ADDITIONS (Part 1):
  - After a valid message is saved, emits 'new_message_sent' socket event
    so the recipient's inbox updates in real-time without a page refresh.
  - Added /api/send JSON endpoint for the socket-aware compose flow.
  - Added /api/online_users endpoint to expose online status.
  - All original routing, spam/vulgar logic, and DB interactions preserved.
"""

import os
import uuid
from functools import wraps
from flask import (
    Blueprint, render_template, request, redirect,
    url_for, session, flash, current_app, jsonify, abort
)
from werkzeug.utils import secure_filename
from database import get_db
from spam_filter import classify_message, is_spam

mail_bp = Blueprint("mail", __name__)


# ─────────────────────────────────────────────────────────────────────────────
# Auth guard
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


def _allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in current_app.config["ALLOWED_EXTENSIONS"]


def _save_attachment(file_obj):
    if not file_obj or file_obj.filename == "":
        return None
    if not _allowed_file(file_obj.filename):
        return None
    safe   = secure_filename(file_obj.filename)
    unique = f"{uuid.uuid4().hex}_{safe}"
    dest   = os.path.join(current_app.config["UPLOAD_FOLDER"], unique)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    file_obj.save(dest)
    return unique


def _get_message_or_404(msg_id, user_id):
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
             AND m.deleted_at IS NULL""",
        (msg_id, user_id, user_id)
    ).fetchone()
    if not msg:
        abort(404)
    return msg


def _do_send_message(sender_id, to_username, subject, body, thread_id, attachment_file):
    """
    Core message-sending logic extracted so it can be called from both
    the traditional form POST and the new JSON API endpoint.

    Returns a dict: { ok, error, recipient_id, message_id, ml, folder }
    """
    db = get_db()

    recipient = db.execute(
        "SELECT * FROM users WHERE username=?", (to_username,)
    ).fetchone()
    if not recipient:
        return {'ok': False, 'error': f'User "{to_username}" not found.'}

    att_filename = _save_attachment(attachment_file)

    # ── ML Moderation Pipeline (unchanged) ────────────────────────────────────
    ml = classify_message(subject, body)

    if ml['blocked']:
        db.execute(
            """INSERT INTO moderation_log
               (sender_id, recipient_id, subject, body,
                vulgar_probability, spam_probability, moderation_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sender_id, recipient["id"], subject, body,
             ml['vulgar_probability'], ml['spam_probability'],
             ml['decision_reason'])
        )
        db.commit()
        return {'ok': False, 'error': ml['warning'], 'blocked': True}

    inbox_folder = ml['folder']
    recipient_id = recipient["id"]

    cur = db.execute(
        """INSERT INTO messages
           (thread_id, sender_id, recipient_id, subject, body, folder,
            attachment,
            spam_flag, spam_probability, ham_probability, spam_confidence,
            vulgar_probability, is_vulgar, is_blocked, moderation_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (thread_id, sender_id, recipient_id, subject, body, inbox_folder,
         att_filename,
         ml['spam_flag'], ml['spam_probability'], ml['ham_probability'],
         ml['confidence'], ml['vulgar_probability'],
         int(ml['is_vulgar']), int(ml['blocked']), ml['decision_reason'])
    )
    recipient_msg_id = cur.lastrowid

    if not thread_id:
        db.execute("UPDATE messages SET thread_id=? WHERE id=?",
                   (recipient_msg_id, recipient_msg_id))

    db.execute(
        """INSERT INTO messages
           (thread_id, sender_id, recipient_id, subject, body, folder,
            is_read, attachment,
            spam_flag, spam_probability, ham_probability, spam_confidence,
            vulgar_probability, is_vulgar, is_blocked, moderation_reason)
           VALUES (?, ?, ?, ?, ?, 'sent', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (thread_id or recipient_msg_id, sender_id, recipient_id,
         subject, body, att_filename,
         ml['spam_flag'], ml['spam_probability'], ml['ham_probability'],
         ml['confidence'], ml['vulgar_probability'],
         int(ml['is_vulgar']), int(ml['blocked']), ml['decision_reason'])
    )

    if ml['is_spam']:
        db.execute(
            "INSERT INTO spam_log (message_id, reason) VALUES (?, ?)",
            (recipient_msg_id, ml['decision_reason'])
        )

    db.commit()
    return {
        'ok':           True,
        'message_id':   recipient_msg_id,
        'recipient_id': recipient_id,
        'ml':           ml,
        'folder':       inbox_folder,
        'recipient':    dict(recipient),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Folder views
# ─────────────────────────────────────────────────────────────────────────────

def _render_folder(folder_name, user_id, template="mailbox.html"):
    db      = get_db()
    page    = request.args.get("page", 1, type=int)
    q       = request.args.get("q", "").strip()
    per_page = current_app.config["MESSAGES_PER_PAGE"]
    offset  = (page - 1) * per_page

    owner_col = "m.sender_id" if folder_name == "sent" else "m.recipient_id"
    search_clause = ""
    params = [user_id, folder_name]

    if q:
        search_clause = " AND (m.subject LIKE ? OR m.body LIKE ?)"
        params += [f"%{q}%", f"%{q}%"]

    params_count = params.copy()
    params += [per_page, offset]

    base_query = f"""
        FROM messages m
        JOIN users s ON s.id = m.sender_id
        JOIN users r ON r.id = m.recipient_id
        WHERE {owner_col}=? AND m.folder=? AND m.deleted_at IS NULL
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
@login_required
def index():
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
# Compose / Send (traditional form POST — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@mail_bp.route("/compose", methods=["GET", "POST"])
@login_required
def compose():
    db = get_db()
    prefill = {
        "to":        request.args.get("to", ""),
        "subject":   request.args.get("subject", ""),
        "body":      request.args.get("body", ""),
        "thread_id": request.args.get("thread_id", ""),
    }

    if request.method == "POST":
        to_username = request.form.get("to", "").strip()
        subject     = request.form.get("subject", "").strip() or "(no subject)"
        body        = request.form.get("body", "").strip()
        thread_id   = request.form.get("thread_id") or None
        attachment  = request.files.get("attachment")

        result = _do_send_message(
            session["user_id"], to_username, subject, body,
            thread_id, attachment
        )

        if not result['ok']:
            flash(f'⚠️ {result["error"]}', "error")
            return render_template(
                "compose.html",
                folder_counts=_folder_counts(session["user_id"]),
                prefill={**prefill, "to": to_username, "subject": subject, "body": body},
                users=db.execute(
                    "SELECT username, display_name FROM users WHERE id != ? ORDER BY username",
                    (session["user_id"],)).fetchall()
            )

        ml = result['ml']
        if ml['is_spam']:
            flash(f'Message sent (auto-filtered to spam: {ml["decision_reason"]}).', "warning")
        else:
            flash("Message sent!", "success")
        return redirect(url_for("mail.sent"))

    users = db.execute(
        "SELECT username, display_name FROM users WHERE id != ? ORDER BY username",
        (session["user_id"],)
    ).fetchall()
    return render_template("compose.html",
                           folder_counts=_folder_counts(session["user_id"]),
                           prefill=prefill, users=users)


# ─────────────────────────────────────────────────────────────────────────────
# View message
# ─────────────────────────────────────────────────────────────────────────────

@mail_bp.route("/message/<int:msg_id>")
@login_required
def view_message(msg_id):
    user_id = session["user_id"]
    db      = get_db()
    msg     = _get_message_or_404(msg_id, user_id)

    if msg["recipient_id"] == user_id and not msg["is_read"]:
        db.execute("UPDATE messages SET is_read=1 WHERE id=?", (msg_id,))
        db.commit()

    thread = db.execute(
        """SELECT m.*,
                  s.username  AS sender_username,
                  s.display_name AS sender_display,
                  s.avatar_color AS sender_color
           FROM messages m
           JOIN users s ON s.id = m.sender_id
           WHERE m.thread_id=? AND m.deleted_at IS NULL
             AND (m.sender_id=? OR m.recipient_id=?)
           ORDER BY m.sent_at ASC""",
        (msg["thread_id"] or msg_id, user_id, user_id)
    ).fetchall()

    return render_template("view_message.html",
                           msg=msg, thread=thread,
                           folder_counts=_folder_counts(user_id))


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────

@mail_bp.route("/message/<int:msg_id>/delete", methods=["POST"])
@login_required
def delete_message(msg_id):
    user_id = session["user_id"]
    db = get_db()
    msg = _get_message_or_404(msg_id, user_id)
    if msg["folder"] == "trash":
        db.execute("DELETE FROM messages WHERE id=?", (msg_id,))
        flash("Message permanently deleted.", "info")
    else:
        db.execute("UPDATE messages SET folder='trash', deleted_at=CURRENT_TIMESTAMP WHERE id=?", (msg_id,))
        flash("Message moved to Trash.", "info")
    db.commit()
    return redirect(request.referrer or url_for("mail.inbox"))


@mail_bp.route("/message/<int:msg_id>/restore", methods=["POST"])
@login_required
def restore_message(msg_id):
    user_id = session["user_id"]
    db = get_db()
    msg = db.execute(
        "SELECT * FROM messages WHERE id=? AND (sender_id=? OR recipient_id=?)",
        (msg_id, user_id, user_id)
    ).fetchone()
    if not msg:
        abort(404)
    restore_to = "sent" if msg["sender_id"] == user_id else "inbox"
    db.execute("UPDATE messages SET folder=?, deleted_at=NULL WHERE id=?", (restore_to, msg_id))
    db.commit()
    flash("Message restored.", "success")
    return redirect(url_for("mail.trash"))


@mail_bp.route("/message/<int:msg_id>/toggle-read", methods=["POST"])
@login_required
def toggle_read(msg_id):
    user_id = session["user_id"]
    db = get_db()
    msg = _get_message_or_404(msg_id, user_id)
    db.execute("UPDATE messages SET is_read=? WHERE id=?", (0 if msg["is_read"] else 1, msg_id))
    db.commit()
    return redirect(request.referrer or url_for("mail.inbox"))


@mail_bp.route("/message/<int:msg_id>/toggle-star", methods=["POST"])
@login_required
def toggle_star(msg_id):
    user_id = session["user_id"]
    db = get_db()
    msg = _get_message_or_404(msg_id, user_id)
    db.execute("UPDATE messages SET is_starred=? WHERE id=?", (0 if msg["is_starred"] else 1, msg_id))
    db.commit()
    return redirect(request.referrer or url_for("mail.inbox"))


@mail_bp.route("/message/<int:msg_id>/mark-spam", methods=["POST"])
@login_required
def mark_spam(msg_id):
    user_id = session["user_id"]
    db = get_db()
    _get_message_or_404(msg_id, user_id)
    db.execute("UPDATE messages SET folder='spam' WHERE id=?", (msg_id,))
    db.commit()
    flash("Message marked as spam.", "warning")
    return redirect(request.referrer or url_for("mail.inbox"))


@mail_bp.route("/message/<int:msg_id>/not-spam", methods=["POST"])
@login_required
def not_spam(msg_id):
    user_id = session["user_id"]
    db = get_db()
    _get_message_or_404(msg_id, user_id)
    db.execute("UPDATE messages SET folder='inbox' WHERE id=?", (msg_id,))
    db.commit()
    flash("Message moved to Inbox.", "success")
    return redirect(request.referrer or url_for("mail.spam"))


@mail_bp.route("/bulk-action", methods=["POST"])
@login_required
def bulk_action():
    user_id = session["user_id"]
    db = get_db()
    ids    = request.form.getlist("msg_ids")
    action = request.form.get("action")

    if not ids or action not in ("delete", "read", "unread", "spam", "star"):
        flash("Invalid bulk action.", "error")
        return redirect(request.referrer or url_for("mail.inbox"))

    placeholders = ",".join("?" for _ in ids)
    ownership    = f"id IN ({placeholders}) AND (sender_id=? OR recipient_id=?)"
    all_params   = ids + [user_id, user_id]

    if action == "delete":
        db.execute(f"UPDATE messages SET folder='trash', deleted_at=CURRENT_TIMESTAMP WHERE {ownership}", all_params)
        flash(f"{len(ids)} message(s) moved to Trash.", "info")
    elif action == "read":
        db.execute(f"UPDATE messages SET is_read=1 WHERE {ownership}", all_params)
    elif action == "unread":
        db.execute(f"UPDATE messages SET is_read=0 WHERE {ownership}", all_params)
    elif action == "spam":
        db.execute(f"UPDATE messages SET folder='spam' WHERE {ownership}", all_params)
        flash(f"{len(ids)} moved to Spam.", "warning")
    elif action == "star":
        db.execute(f"UPDATE messages SET is_starred=1 WHERE {ownership}", all_params)

    db.commit()
    return redirect(request.referrer or url_for("mail.inbox"))


# ─────────────────────────────────────────────────────────────────────────────
# Profile
# ─────────────────────────────────────────────────────────────────────────────

@mail_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    db = get_db()
    user_id = session["user_id"]
    user = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()

    if request.method == "POST":
        display_name = request.form.get("display_name", "").strip() or user["username"]
        avatar_color = request.form.get("avatar_color", user["avatar_color"])
        db.execute("UPDATE users SET display_name=?, avatar_color=? WHERE id=?",
                   (display_name, avatar_color, user_id))
        db.commit()
        session["display_name"] = display_name
        session["avatar_color"] = avatar_color
        flash("Profile updated.", "success")
        return redirect(url_for("mail.profile"))

    stats = db.execute(
        """SELECT
             SUM(CASE WHEN folder='inbox' AND recipient_id=? THEN 1 ELSE 0 END) AS inbox_count,
             SUM(CASE WHEN folder='sent'  AND sender_id=?    THEN 1 ELSE 0 END) AS sent_count,
             SUM(CASE WHEN folder='spam'  AND recipient_id=? THEN 1 ELSE 0 END) AS spam_count
           FROM messages WHERE deleted_at IS NULL""",
        (user_id, user_id, user_id)
    ).fetchone()

    return render_template("profile.html", user=user, stats=stats,
                           folder_counts=_folder_counts(user_id))


# ─────────────────────────────────────────────────────────────────────────────
# REST API  (existing + new socket-related endpoints)
# ─────────────────────────────────────────────────────────────────────────────

@mail_bp.route("/api/users/search")
@login_required
def api_user_search():
    q  = request.args.get("q", "").strip()
    db = get_db()
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


# ── NEW: JSON send endpoint used by socket-aware compose ─────────────────────
@mail_bp.route("/api/send", methods=["POST"])
@login_required
def api_send():
    """
    SOCKET INTEGRATION (Part 1):
    JSON endpoint called by the compose page after a message is sent.
    Returns message metadata so the client can emit 'new_message_sent'
    to the SocketIO server, which then pushes 'new_message' to the
    recipient's browser room in real-time.
    """
    data        = request.get_json(force=True) or {}
    to_username = data.get("to", "").strip()
    subject     = data.get("subject", "").strip() or "(no subject)"
    body        = data.get("body", "").strip()
    thread_id   = data.get("thread_id") or None

    result = _do_send_message(
        session["user_id"], to_username, subject, body, thread_id, None
    )

    if not result['ok']:
        return jsonify({'ok': False, 'error': result['error']}), 400

    db = get_db()
    sender = db.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()

    return jsonify({
        'ok':             True,
        'message_id':     result['message_id'],
        'recipient_id':   result['recipient_id'],
        'subject':        subject,
        'preview':        body[:80],
        'sender_display': sender["display_name"] or sender["username"],
        'sender_color':   sender["avatar_color"] or "#6c63ff",
        'is_spam':        result['ml']['is_spam'],
        'folder':         result['folder'],
    })


# ── NEW: Online user list API ─────────────────────────────────────────────────
@mail_bp.route("/api/online-users")
@login_required
def api_online_users():
    """
    SOCKET INTEGRATION (Part 1 — Online/Offline Status):
    Returns the list of currently online user IDs.
    Polled on page load; real-time updates come via the 'online_users' socket event.
    """
    from app import _online_users
    return jsonify({'online': list(set(_online_users.values()))})
