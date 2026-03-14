"""
app.py - Flask application entry point (ENHANCED)
===================================================
CHANGES:
  - Integrated Flask-SocketIO for real-time messaging (Part 1)
  - Added online/offline user tracking via socket events
  - Registered model_stats blueprint for ML dashboard (Part 2)
  - All existing blueprints and models unchanged
"""

import os
from flask import Flask, request as flask_request, session as flask_session
from flask_socketio import SocketIO, emit, join_room, leave_room
from config import Config
from database import init_db
from routes.auth import auth_bp
from routes.mail import mail_bp
from routes.stats import stats_bp          # NEW: ML stats dashboard
from naive_bayes import NaiveBayesSpamClassifier
from vulgar_classifier import VulgarClassifier

# ── SocketIO instance (shared across modules) ─────────────────────────────────
# async_mode='threading' works with the built-in dev server; no extra packages needed.
socketio = SocketIO(async_mode='threading', cors_allowed_origins='*')

# ── In-memory online user registry ───────────────────────────────────────────
# Maps socket session id (sid) → user_id
_online_users: dict = {}   # sid -> user_id


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    init_db(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(mail_bp)
    app.register_blueprint(stats_bp)       # NEW: /model_stats

    # ── Load & Train models (unchanged) ──────────────────────────────────────
    app.nb_model = NaiveBayesSpamClassifier()
    app.nb_model.train("dataset/SpamCollection.txt")

    app.vulgar_model = VulgarClassifier(alpha=1.0, vulgar_threshold=0.60)
    app.vulgar_model.train("dataset/vulgar_dataset.txt")

    # ── Attach SocketIO ───────────────────────────────────────────────────────
    socketio.init_app(app)

    @app.errorhandler(404)
    def page_not_found(e):
        return "<h2 style='font-family:monospace;padding:2rem'>404 — Page not found</h2>", 404

    @app.errorhandler(500)
    def server_error(e):
        return "<h2 style='font-family:monospace;padding:2rem'>500 — Server error</h2>", 500

    return app


app = create_app()


# ═════════════════════════════════════════════════════════════════════════════
# SOCKET EVENT HANDLERS  (Part 1 — Real-Time Communication)
# ═════════════════════════════════════════════════════════════════════════════

@socketio.on('connect')
def handle_connect():
    """
    SOCKET EVENT: connect
    Reads user_id from Flask session, registers the user as online,
    and broadcasts the updated online-user list to all clients.
    """
    user_id = flask_session.get('user_id')
    if user_id:
        _online_users[flask_request.sid] = user_id
        join_room(f"user_{user_id}")   # personal room for targeted delivery
        emit('online_users', {'users': list(set(_online_users.values()))}, broadcast=True)
        print(f"[Socket] User {user_id} connected (sid={flask_request.sid})")


@socketio.on('disconnect')
def handle_disconnect():
    """
    SOCKET EVENT: disconnect
    Removes user from registry and broadcasts updated online list.
    """
    user_id = _online_users.pop(flask_request.sid, None)
    if user_id:
        leave_room(f"user_{user_id}")
        emit('online_users', {'users': list(set(_online_users.values()))}, broadcast=True)
        print(f"[Socket] User {user_id} disconnected (sid={flask_request.sid})")


@socketio.on('typing')
def handle_typing(data):
    """
    SOCKET EVENT: typing
    Client emits: { recipient_id, username, is_typing }
    Forwards typing indicator to the target recipient's personal room.
    """
    recipient_id = data.get('recipient_id')
    if recipient_id:
        emit('typing', {
            'username':  data.get('username', 'Someone'),
            'is_typing': data.get('is_typing', False)
        }, room=f"user_{recipient_id}")


@socketio.on('new_message_sent')
def handle_new_message(data):
    """
    SOCKET EVENT: new_message_sent
    Client emits after a successful POST /compose:
      { recipient_id, message_id, subject, preview, sender_display, sender_color, sent_at }
    Server pushes 'new_message' to the recipient's room to trigger
    live inbox update and browser notification.
    """
    recipient_id = data.get('recipient_id')
    if recipient_id:
        emit('new_message', {
            'message_id':     data.get('message_id'),
            'subject':        data.get('subject', '(no subject)'),
            'preview':        data.get('preview', ''),
            'sender_display': data.get('sender_display', 'Someone'),
            'sender_color':   data.get('sender_color', '#6c63ff'),
            'sent_at':        data.get('sent_at', ''),
        }, room=f"user_{recipient_id}")


if __name__ == "__main__":
    # IMPORTANT: Use socketio.run() — not app.run() — to enable WebSocket support
    socketio.run(app, debug=True, host="127.0.0.1", port=5000)
