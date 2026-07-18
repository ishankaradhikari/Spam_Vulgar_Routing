"""
app.py - Flask application entry point
Run with: python app.py  (or  flask run)
"""

import os
from flask import Flask
from config import Config
from database import init_db
from routes.auth import auth_bp
from routes.mail import mail_bp
from naive_bayes import NaiveBayesSpamClassifier
from ml_settings import VULGAR_ALPHA, VULGAR_THRESHOLD

# NEW: import the vulgar classifier (does NOT modify NaiveBayesSpamClassifier)
from vulgar_classifier import VulgarClassifier


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # ── Initialise database (creates tables + runs migration) ─────────────────
    init_db(app)

    # ── Register blueprints ───────────────────────────────────────────────────
    app.register_blueprint(auth_bp)
    app.register_blueprint(mail_bp)
    # Note: ML stats web route removed - use CLI `scripts/print_ml_stats.py` for reports

    # ── Load & Train Naive Bayes Spam Model (UNCHANGED) ───────────────────────
    # This is the existing model - we do NOT modify it.
    app.nb_model = NaiveBayesSpamClassifier()
    spam_dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "SpamCollection.txt")
    app.nb_model.train(spam_dataset_path)

    # ── Load & Train Vulgar Language Classifier (NEW) ─────────────────────────
    # A second, independent classifier that detects vulgar/abusive language.
    # It uses character n-grams + word features + special-char ratios so it
    # can detect obfuscated words like bi*ch, @ss, f!ck, etc.
    app.vulgar_model = VulgarClassifier(alpha=VULGAR_ALPHA, vulgar_threshold=VULGAR_THRESHOLD)
    vulgar_dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "vulgar_dataset.txt")

    # Train vulgar model at startup (in-memory). We intentionally do NOT
    # persist a cached pickle to avoid duplicated writes and stale cache issues.
    app.vulgar_model.train(vulgar_dataset_path)
    


    

    # ── Custom 404 / 500 pages ────────────────────────────────────────────────
    @app.errorhandler(404)
    def page_not_found(e):
        return "<h2 style='font-family:monospace;padding:2rem'>404 - Page not found</h2>", 404

    @app.errorhandler(500)
    def server_error(e):
        return "<h2 style='font-family:monospace;padding:2rem'>500 - Server error</h2>", 500


    return app


app = create_app()

if __name__ == "__main__":
    app.run(use_reloader=False, host="127.0.0.1", port=5000)
