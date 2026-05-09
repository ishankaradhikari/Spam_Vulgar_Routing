"""
routes/stats.py — ML Statistics Dashboard
====================================================
Exposes GET /model_stats which evaluates the Naive Bayes spam classifier
on the same dataset it was trained on (train-set evaluation) and displays:
  - Confusion matrix  (TP, TN, FP, FN)
  - Accuracy
  - Precision
  - Recall
  - F1 Score
"""

import math
import os
from functools import wraps
from flask import Blueprint, render_template, current_app, session, redirect, url_for, flash
from database import get_db

stats_bp = Blueprint("stats", __name__)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "info")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


def _get_spam_probability(model, message: str) -> float:
    words = model.tokenize(message)
    total_msgs = model.spam_messages + model.ham_messages
    if total_msgs == 0:
        return 0.5

    log_spam = math.log(model.spam_messages / total_msgs)
    log_ham = math.log(model.ham_messages / total_msgs)

    for word in words:
        if word in model.STOP_WORDS:
            continue
        log_spam += math.log(model.word_prob(word, 'spam'))
        log_ham += math.log(model.word_prob(word, 'ham'))

    max_log = max(log_spam, log_ham)
    exp_spam = math.exp(log_spam - max_log)
    exp_ham = math.exp(log_ham - max_log)
    return exp_spam / (exp_spam + exp_ham)


def _get_vulgar_probability(model, message: str) -> float:
    proba = model.predict_proba(message)
    return proba[model.VULGAR]


def compute_stats(dataset_path: str, threshold: float = 0.70) -> dict:
    model = current_app.nb_model
    TP = TN = FP = FN = 0
    total = 0
    examples = []

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue

                true_label, text = parts[0].lower(), parts[1]
                spam_prob = _get_spam_probability(model, text)
                pred_label = "spam" if spam_prob >= threshold else "ham"

                total += 1
                if true_label == "spam" and pred_label == "spam":
                    TP += 1
                elif true_label == "ham" and pred_label == "ham":
                    TN += 1
                elif true_label == "ham" and pred_label == "spam":
                    FP += 1
                elif true_label == "spam" and pred_label == "ham":
                    FN += 1

                if total <= 200:
                    examples.append({
                        'true': true_label,
                        'pred': pred_label,
                        'prob': round(spam_prob * 100, 1),
                        'correct': true_label == pred_label,
                        'text': text[:80],
                    })

    except FileNotFoundError:
        return {'error': f'Dataset not found: {dataset_path}'}

    denom_acc = TP + TN + FP + FN
    denom_prec = TP + FP
    denom_rec = TP + FN

    accuracy = round((TP + TN) / denom_acc * 100, 2) if denom_acc else 0.0
    precision = round(TP / denom_prec * 100, 2) if denom_prec else 0.0
    recall = round(TP / denom_rec * 100, 2) if denom_rec else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) else 0.0

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'total': total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'spam_count': model.spam_messages,
        'ham_count': model.ham_messages,
        'vocab_size': len(model.vocabulary),
        'examples': examples,
    }


def compute_vulgar_stats(dataset_path: str, threshold: float = 0.90) -> dict:
    model = current_app.vulgar_model
    TP = TN = FP = FN = 0
    total = 0
    examples = []

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Support both tab-separated and space-separated
                if '\t' in line:
                    parts = line.split('\t', 1)
                else:
                    parts = line.split(None, 1)
                
                if len(parts) < 2:
                    continue

                true_label, text = parts[0].lower(), parts[1]
                # Normalize label variants
                if true_label == 'vulgar':
                    true_label = 'vulgar'
                else:
                    true_label = 'clean'
                
                vulgar_prob = _get_vulgar_probability(model, text)
                pred_label = "vulgar" if vulgar_prob >= threshold else "clean"

                total += 1
                if true_label == "vulgar" and pred_label == "vulgar":
                    TP += 1
                elif true_label == "clean" and pred_label == "clean":
                    TN += 1
                elif true_label == "clean" and pred_label == "vulgar":
                    FP += 1
                elif true_label == "vulgar" and pred_label == "clean":
                    FN += 1

                if total <= 200:
                    examples.append({
                        'true': true_label,
                        'pred': pred_label,
                        'prob': round(vulgar_prob * 100, 1),
                        'correct': true_label == pred_label,
                        'text': text[:80],
                    })

    except FileNotFoundError:
        return {'error': f'Dataset not found: {dataset_path}'}

    denom_acc = TP + TN + FP + FN
    denom_prec = TP + FP
    denom_rec = TP + FN

    accuracy = round((TP + TN) / denom_acc * 100, 2) if denom_acc else 0.0
    precision = round(TP / denom_prec * 100, 2) if denom_prec else 0.0
    recall = round(TP / denom_rec * 100, 2) if denom_rec else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) else 0.0

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'total': total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'vulgar_count': model.class_counts[model.VULGAR],
        'clean_count': model.class_counts[model.CLEAN],
        'vocab_size': len(model.vocabulary),
        'examples': examples,
    }


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
        counts[r['folder']] = r['cnt']
    return counts


@stats_bp.route("/model_stats")
@login_required
def model_stats():
    spam_dataset_path = os.path.join(current_app.root_path, "dataset", "SpamCollection.txt")
    vulgar_dataset_path = os.path.join(current_app.root_path, "dataset", "vulgar_dataset.txt")
    
    spam_stats = compute_stats(spam_dataset_path)
    vulgar_stats = compute_vulgar_stats(vulgar_dataset_path)
    
    return render_template("model_stats.html",
                           spam_stats=spam_stats,
                           vulgar_stats=vulgar_stats,
                           folder_counts=_folder_counts(session["user_id"]))
