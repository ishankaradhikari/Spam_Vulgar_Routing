"""
routes/stats.py — ML Statistics Dashboard (Part 2)
====================================================
Exposes GET /model_stats which evaluates the Naive Bayes spam classifier
on the same dataset it was trained on (train-set evaluation) and displays:
  - Confusion matrix  (TP, TN, FP, FN)
  - Accuracy
  - Precision
  - Recall
  - F1 Score

HOW THE METRICS ARE CALCULATED
--------------------------------
We run every labelled example in SpamCollection.txt through the trained
NaiveBayesSpamClassifier using the same threshold logic as spam_filter.py
(spam_probability >= 0.70 → SPAM, else HAM).

  TP (True Positive)  : model said SPAM  and label is spam
  TN (True Negative)  : model said HAM   and label is ham
  FP (False Positive) : model said SPAM  but label is ham  (false alarm)
  FN (False Negative) : model said HAM   but label is spam (missed spam)

  Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  Precision = TP / (TP + FP)   — of predicted spam, how many are truly spam
  Recall    = TP / (TP + FN)   — of all actual spam, how many did we catch
  F1 Score  = 2 * (Precision * Recall) / (Precision + Recall)

NOTE: Train-set evaluation inflates all metrics because the model has
already seen these examples. For production, split into train/test sets.
"""

import math
from flask import Blueprint, render_template, current_app
from functools import wraps
from flask import session, redirect, url_for, flash

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
    """
    Replicates spam_filter._get_spam_probability() without importing it
    to keep this module self-contained.
    Returns P(spam | message) in [0, 1].
    """
    words = model.tokenize(message)
    total_msgs = model.spam_messages + model.ham_messages
    if total_msgs == 0:
        return 0.5

    log_spam = math.log(model.spam_messages / total_msgs)
    log_ham  = math.log(model.ham_messages  / total_msgs)

    for word in words:
        if word in model.STOP_WORDS:
            continue
        log_spam += math.log(model.word_prob(word, 'spam'))
        log_ham  += math.log(model.word_prob(word, 'ham'))

    max_log   = max(log_spam, log_ham)
    exp_spam  = math.exp(log_spam - max_log)
    exp_ham   = math.exp(log_ham  - max_log)
    return exp_spam / (exp_spam + exp_ham)


def compute_stats(dataset_path: str, threshold: float = 0.70) -> dict:
    """
    Evaluates the trained NB model on dataset_path and returns a dict
    containing the confusion matrix values and derived metrics.
    """
    model = current_app.nb_model
    TP = TN = FP = FN = 0
    total = 0
    examples = []   # list of (true_label, predicted_label, prob, text_snippet)

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

                if total <= 200:   # keep first 200 for sample table
                    examples.append({
                        'true':    true_label,
                        'pred':    pred_label,
                        'prob':    round(spam_prob * 100, 1),
                        'correct': true_label == pred_label,
                        'text':    text[:80],
                    })

    except FileNotFoundError:
        return {'error': f'Dataset not found: {dataset_path}'}

    # ── Derived metrics ───────────────────────────────────────────────────────
    denom_acc  = TP + TN + FP + FN
    denom_prec = TP + FP
    denom_rec  = TP + FN

    accuracy  = round((TP + TN) / denom_acc  * 100, 2) if denom_acc  else 0.0
    precision = round(TP / denom_prec         * 100, 2) if denom_prec else 0.0
    recall    = round(TP / denom_rec          * 100, 2) if denom_rec  else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 2) \
                if (precision + recall) else 0.0

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'total':     total,
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'threshold': threshold,
        'spam_count':  model.spam_messages,
        'ham_count':   model.ham_messages,
        'vocab_size':  len(model.vocabulary),
        'examples':    examples,
    }


@stats_bp.route("/model_stats")
@login_required
def model_stats():
    stats = compute_stats("dataset/SpamCollection.txt")
    return render_template("model_stats.html", stats=stats)
