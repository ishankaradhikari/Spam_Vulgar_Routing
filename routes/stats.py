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
import random
import tempfile
from functools import wraps
from flask import Blueprint, render_template, current_app, session, redirect, url_for, flash
from database import get_db
from naive_bayes import NaiveBayesSpamClassifier
from vulgar_classifier import VulgarClassifier

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


def load_labeled_examples(dataset_path: str) -> list[tuple[str, str]]:
    examples = []

    with open(dataset_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if '\t' in line:
                parts = line.split('\t', 1)
            else:
                parts = line.split(None, 1)

            if len(parts) < 2:
                continue

            label = parts[0].strip().lower()
            text = parts[1].strip()
            examples.append((label, text))

    return examples


def split_dataset_examples(examples: list[tuple[str, str]], train_ratio: float = 0.8, random_seed: int = 42) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not examples:
        return [], []

    shuffled = list(examples)
    rng = random.Random(random_seed)
    rng.shuffle(shuffled)

    split_index = int(len(shuffled) * train_ratio)
    split_index = max(1, min(split_index, len(shuffled) - 1))
    return shuffled[:split_index], shuffled[split_index:]


def _train_model_from_examples(model, examples: list[tuple[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        for label, text in examples:
            handle.write(f"{label}\t{text}\n")
        temp_path = handle.name

    try:
        model.train(temp_path)
    finally:
        os.unlink(temp_path)


def _evaluate_spam_examples(model, examples: list[tuple[str, str]], threshold: float = 0.70) -> dict:
    TP = TN = FP = FN = 0
    total = 0
    sample_examples = []

    for true_label, text in examples:
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
            sample_examples.append({
                'true': true_label,
                'pred': pred_label,
                'prob': round(spam_prob * 100, 1),
                'correct': true_label == pred_label,
                'text': text[:80],
            })

    denom_acc = TP + TN + FP + FN
    denom_prec = TP + FP
    denom_rec = TP + FN

    accuracy = round((TP + TN) / denom_acc * 100, 2) if denom_acc else 0.0
    precision = round(TP / denom_prec * 100, 2) if denom_prec else 0.0
    recall = round(TP / denom_rec * 100, 2) if denom_rec else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) else 0.0

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'total': total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'spam_count': model.spam_messages,
        'ham_count': model.ham_messages,
        'vocab_size': len(model.vocabulary),
        'examples': sample_examples,
    }


def _evaluate_vulgar_examples(model, examples: list[tuple[str, str]], threshold: float = 0.90) -> dict:
    TP = TN = FP = FN = 0
    total = 0
    sample_examples = []

    for true_label, text in examples:
        true_label = 'vulgar' if true_label == 'vulgar' else 'clean'
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
            sample_examples.append({
                'true': true_label,
                'pred': pred_label,
                'prob': round(vulgar_prob * 100, 1),
                'correct': true_label == pred_label,
                'text': text[:80],
            })

    denom_acc = TP + TN + FP + FN
    denom_prec = TP + FP
    denom_rec = TP + FN

    accuracy = round((TP + TN) / denom_acc * 100, 2) if denom_acc else 0.0
    precision = round(TP / denom_prec * 100, 2) if denom_prec else 0.0
    recall = round(TP / denom_rec * 100, 2) if denom_rec else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) else 0.0

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'total': total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'vulgar_count': model.class_counts[model.VULGAR],
        'clean_count': model.class_counts[model.CLEAN],
        'vocab_size': len(model.vocabulary),
        'examples': sample_examples,
    }


def compute_stats(dataset_path: str, threshold: float = 0.70) -> dict:
    try:
        examples = load_labeled_examples(dataset_path)
    except FileNotFoundError:
        return {'error': f'Dataset not found: {dataset_path}'}

    train_examples, test_examples = split_dataset_examples(examples, train_ratio=0.8, random_seed=42)
    train_model = NaiveBayesSpamClassifier()
    _train_model_from_examples(train_model, train_examples)

    return {
        'train': _evaluate_spam_examples(train_model, train_examples, threshold=threshold),
        'test': _evaluate_spam_examples(train_model, test_examples, threshold=threshold),
        'train_ratio': 0.8,
        'test_ratio': 0.2,
        'train_size': len(train_examples),
        'test_size': len(test_examples),
    }


def compute_vulgar_stats(dataset_path: str, threshold: float = 0.90) -> dict:
    try:
        examples = load_labeled_examples(dataset_path)
    except FileNotFoundError:
        return {'error': f'Dataset not found: {dataset_path}'}

    train_examples, test_examples = split_dataset_examples(examples, train_ratio=0.8, random_seed=42)
    train_model = VulgarClassifier(alpha=2.0, vulgar_threshold=threshold)
    _train_model_from_examples(train_model, train_examples)

    return {
        'train': _evaluate_vulgar_examples(train_model, train_examples, threshold=threshold),
        'test': _evaluate_vulgar_examples(train_model, test_examples, threshold=threshold),
        'train_ratio': 0.8,
        'test_ratio': 0.2,
        'train_size': len(train_examples),
        'test_size': len(test_examples),
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
