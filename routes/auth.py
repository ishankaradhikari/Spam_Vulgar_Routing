"""
routes/auth.py - Authentication blueprint (register, login, logout)
"""

import random
from flask import (
    Blueprint, render_template, request, redirect,
    url_for, session, flash, current_app
)
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db

auth_bp = Blueprint("auth", __name__)

# Palette for auto-assigned avatar colours
AVATAR_COLORS = [
    "#e63946", "#2a9d8f", "#e9c46a", "#f4a261",
    "#264653", "#6c63ff", "#48cae4", "#b5838d",
]


def _validate_registration(username, email, password, confirm):
    errors = []
    if not username or len(username) < 3:
        errors.append("Username must be at least 3 characters.")
    if not email or "@" not in email:
        errors.append("Valid email required.")
    if not password or len(password) < 6:
        errors.append("Password must be at least 6 characters.")
    if password != confirm:
        errors.append("Passwords do not match.")
    return errors


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("mail.inbox"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        errors = _validate_registration(username, email, password, confirm)
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("register.html", username=username, email=email)

        db = get_db()
        existing = db.execute(
            "SELECT id FROM users WHERE username=? OR email=?", (username, email)
        ).fetchone()
        if existing:
            flash("Username or email already taken.", "error")
            return render_template("register.html", username=username, email=email)

        color = random.choice(AVATAR_COLORS)
        db.execute(
            """INSERT INTO users (username, email, password_hash, display_name, avatar_color)
               VALUES (?, ?, ?, ?, ?)""",
            (username, email, generate_password_hash(password), username, color)
        )
        db.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("mail.inbox"))

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password   = request.form.get("password", "")

        db   = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username=? OR email=?",
            (identifier, identifier)
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"]      = user["id"]
            session["username"]     = user["username"]
            session["display_name"] = user["display_name"] or user["username"]
            session["avatar_color"] = user["avatar_color"]
            # Update last_login
            db.execute("UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE id=?", (user["id"],))
            db.commit()
            return redirect(url_for("mail.inbox"))

        flash("Invalid credentials.", "error")

    return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("You've been logged out.", "info")
    return redirect(url_for("auth.login"))
