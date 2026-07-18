"""Shared ML utilities: dataset loading, splitting, training helper, and spam probability.

This centralises common code used by routes, scripts, and the spam pipeline
to avoid duplication and keep behaviour consistent.
"""
import os
import random
import math
import tempfile


def load_labeled_examples(dataset_path: str):
    examples = []
    with open(dataset_path, 'r', encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
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


def split_dataset_examples(examples, train_ratio=0.8, random_seed=42):
    if not examples:
        return [], []
    shuffled = list(examples)
    rng = random.Random(random_seed)
    rng.shuffle(shuffled)
    split_index = int(len(shuffled) * train_ratio)
    split_index = max(1, min(split_index, len(shuffled) - 1))
    return shuffled[:split_index], shuffled[split_index:]


def train_model_from_examples(model, examples):
    # Write examples to a temp file and call the model's train() which
    # expects a dataset file path.
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
        for label, text in examples:
            handle.write(f"{label}\t{text}\n")
        temp_path = handle.name
    try:
        model.train(temp_path)
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def get_spam_probability(model, message: str) -> float:
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


def classification_report_from_confusion(TP: int, TN: int, FP: int, FN: int, positive_label: str = 'pos', negative_label: str = 'neg') -> dict:
    """Return a sklearn-like classification report and aggregate averages for binary confusion counts.

    Returns a dict with per-class precision/recall/f1/support plus accuracy, macro avg and weighted avg.
    """
    total = TP + TN + FP + FN

    support_pos = TP + FN
    support_neg = TN + FP

    # positive class metrics
    prec_pos = TP / (TP + FP) if (TP + FP) else 0.0
    rec_pos = TP / (TP + FN) if (TP + FN) else 0.0
    f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) else 0.0

    # negative class metrics (treat 'neg' as positive for calculations)
    prec_neg = TN / (TN + FN) if (TN + FN) else 0.0
    rec_neg = TN / (TN + FP) if (TN + FP) else 0.0
    f1_neg = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) else 0.0

    accuracy = (TP + TN) / total if total else 0.0

    # macro averages
    macro_precision = (prec_pos + prec_neg) / 2
    macro_recall = (rec_pos + rec_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    # weighted averages
    weighted_precision = (prec_pos * support_pos + prec_neg * support_neg) / (support_pos + support_neg) if (support_pos + support_neg) else 0.0
    weighted_recall = (rec_pos * support_pos + rec_neg * support_neg) / (support_pos + support_neg) if (support_pos + support_neg) else 0.0
    weighted_f1 = (f1_pos * support_pos + f1_neg * support_neg) / (support_pos + support_neg) if (support_pos + support_neg) else 0.0

    report = {
        positive_label: {
            'precision': round(prec_pos, 4),
            'recall': round(rec_pos, 4),
            'f1-score': round(f1_pos, 4),
            'support': support_pos,
        },
        negative_label: {
            'precision': round(prec_neg, 4),
            'recall': round(rec_neg, 4),
            'f1-score': round(f1_neg, 4),
            'support': support_neg,
        },
        'accuracy': round(accuracy, 4),
        'macro avg': {
            'precision': round(macro_precision, 4),
            'recall': round(macro_recall, 4),
            'f1-score': round(macro_f1, 4),
            'support': support_pos + support_neg,
        },
        'weighted avg': {
            'precision': round(weighted_precision, 4),
            'recall': round(weighted_recall, 4),
            'f1-score': round(weighted_f1, 4),
            'support': support_pos + support_neg,
        }
    }

    return report
