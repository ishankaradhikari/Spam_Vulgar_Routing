#!/usr/bin/env python3
"""Print ML train/test stats for spam and vulgar datasets.

Usage: python scripts/print_ml_stats.py

This script preserves the existing evaluation logic (TP/TN/FP/FN, accuracy,
precision, recall, F1) and only changes the terminal formatting to produce a
professional, sklearn-like report and confusion matrix for both TRAIN and TEST
splits of the Spam and Vulgar datasets.
"""
import os
import sys
import math

# Ensure project root is importable when this script is run from `scripts/`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from naive_bayes import NaiveBayesSpamClassifier
from ml_settings import SPAM_THRESHOLD, VULGAR_ALPHA, VULGAR_THRESHOLD
from vulgar_classifier import VulgarClassifier
from ml_utils import load_labeled_examples, split_dataset_examples, train_model_from_examples, get_spam_probability

# ---------------------------------------------------------------------
# Existing evaluation (kept exactly as before)
# ---------------------------------------------------------------------
def evaluate_spam_examples(model, examples, threshold=SPAM_THRESHOLD):
    TP = TN = FP = FN = 0
    total = 0
    for true_label, text in examples:
        spam_prob = get_spam_probability(model, text)
        pred_label = 'spam' if spam_prob >= threshold else 'ham'
        total += 1
        if true_label == 'spam' and pred_label == 'spam':
            TP += 1
        elif true_label == 'ham' and pred_label == 'ham':
            TN += 1
        elif true_label == 'ham' and pred_label == 'spam':
            FP += 1
        elif true_label == 'spam' and pred_label == 'ham':
            FN += 1

    denom_acc = TP + TN + FP + FN
    denom_prec = TP + FP
    denom_rec = TP + FN
    accuracy = (TP + TN) / denom_acc if denom_acc else 0.0
    precision = TP / denom_prec if denom_prec else 0.0
    recall = TP / denom_rec if denom_rec else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'total': total,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1
    }


def evaluate_vulgar_examples(model, examples, threshold=VULGAR_THRESHOLD):
    TP = TN = FP = FN = 0
    total = 0
    for true_label, text in examples:
        true_label = 'vulgar' if true_label == 'vulgar' else 'clean'
        proba = model.predict_proba(text)
        vulgar_prob = proba[model.VULGAR]
        pred_label = 'vulgar' if vulgar_prob >= threshold else 'clean'
        total += 1
        if true_label == 'vulgar' and pred_label == 'vulgar':
            TP += 1
        elif true_label == 'clean' and pred_label == 'clean':
            TN += 1
        elif true_label == 'clean' and pred_label == 'vulgar':
            FP += 1
        elif true_label == 'vulgar' and pred_label == 'clean':
            FN += 1

    denom_acc = TP + TN + FP + FN
    denom_prec = TP + FP
    denom_rec = TP + FN
    accuracy = (TP + TN) / denom_acc if denom_acc else 0.0
    precision = TP / denom_prec if denom_prec else 0.0
    recall = TP / denom_rec if denom_rec else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'total': total,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1
    }

# ---------------------------------------------------------------------
# Helper: build report numbers from confusion counts using same formulas
# (keeps math identical to evaluate_* above)
# ---------------------------------------------------------------------
def build_report_from_counts(TP, TN, FP, FN, pos_label, neg_label):
    support_pos = TP + FN
    support_neg = TN + FP
    total = support_pos + support_neg

    # positive class metrics
    prec_pos = TP / (TP + FP) if (TP + FP) else 0.0
    rec_pos = TP / (TP + FN) if (TP + FN) else 0.0
    f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) else 0.0

    # negative class metrics (treat negative as 'positive' for its counts)
    prec_neg = TN / (TN + FN) if (TN + FN) else 0.0
    rec_neg = TN / (TN + FP) if (TN + FP) else 0.0
    f1_neg = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) else 0.0

    accuracy = (TP + TN) / total if total else 0.0

    # macro averages
    macro_precision = (prec_pos + prec_neg) / 2
    macro_recall = (rec_pos + rec_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    # weighted averages
    weighted_precision = (prec_pos * support_pos + prec_neg * support_neg) / total if total else 0.0
    weighted_recall = (rec_pos * support_pos + rec_neg * support_neg) / total if total else 0.0
    weighted_f1 = (f1_pos * support_pos + f1_neg * support_neg) / total if total else 0.0

    return {
        pos_label: {
            'precision': prec_pos,
            'recall': rec_pos,
            'f1-score': f1_pos,
            'support': support_pos
        },
        neg_label: {
            'precision': prec_neg,
            'recall': rec_neg,
            'f1-score': f1_neg,
            'support': support_neg
        },
        'accuracy': accuracy,
        'macro avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1-score': macro_f1,
            'support': total
        },
        'weighted avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1-score': weighted_f1,
            'support': total
        }
    }

# ---------------------------------------------------------------------
# Pretty-print helpers (fixed-width formatting)
# ---------------------------------------------------------------------
def pct(x):
    """Format fraction as percentage string with two decimals (e.g. 94.14%)."""
    return f"{x * 100:6.2f}%"

def print_dataset_summary(title, examples, vocab_size, train_size, test_size, positive_label, negative_label):
    pos_count = sum(1 for lbl, _ in examples if lbl.lower() == positive_label.lower())
    neg_count = sum(1 for lbl, _ in examples if lbl.lower() == negative_label.lower())
    width = 55
    print("=" * width)
    print(f"{title}")
    print("=" * width)
    print()
    print(f"{positive_label:22s}: {pos_count}")
    print(f"{negative_label:22s}: {neg_count}")
    print()
    print(f"{'Vocabulary Size':22s}: {vocab_size}")
    print()
    print(f"{'Training samples':22s}: {train_size}")
    print(f"{'Testing samples':22s}: {test_size}")
    print()
    print("=" * width)
    print()

def print_confusion_matrix(confusion, pos_label, neg_label, header_title):
    # confusion: dict with TP, TN, FP, FN (TP = actual pos predicted pos)
    TP = confusion['TP']
    TN = confusion['TN']
    FP = confusion['FP']
    FN = confusion['FN']

    actual_pos = f"Actual {pos_label}"
    actual_neg = f"Actual {neg_label}"
    row_label_w = max(len(actual_pos), len(actual_neg))
    num_w = max(
        len(str(TP)), len(str(TN)), len(str(FP)), len(str(FN)),
        len(pos_label), len(neg_label), 6
    )
    gap = 3
    data_width = num_w * 2 + gap
    width = max(row_label_w + gap + data_width, len(header_title))

    print("=" * width)
    print(f"{header_title}")
    print("=" * width)
    print()

    print(f"{'':{row_label_w + gap}}{'Predicted':^{data_width}}")
    print(f"{'':{row_label_w + gap}}{pos_label:>{num_w}}{'':{gap}}{neg_label:>{num_w}}")
    print(f"{actual_pos:<{row_label_w}}{'':{gap}}{TP:>{num_w}}{'':{gap}}{FN:>{num_w}}")
    print(f"{actual_neg:<{row_label_w}}{'':{gap}}{FP:>{num_w}}{'':{gap}}{TN:>{num_w}}")
    print()
    print("-" * width)
    print()


def print_classification_report(report, title):
    # report: dict as built by build_report_from_counts
    class_names = [k for k in report.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
    class_names += ['Accuracy', 'Macro Avg', 'Weighted Avg']

    cls_col = max(len('Class'), max(len(k) for k in class_names))
    prec_col = max(
        len('Precision'),
        max(len(pct(report[k]['precision'])) for k in report if k not in ('accuracy', 'macro avg', 'weighted avg'))
    )
    rec_col = max(
        len('Recall'),
        max(len(pct(report[k]['recall'])) for k in report if k not in ('accuracy', 'macro avg', 'weighted avg'))
    )
    f1_col = max(
        len('F1-Score'),
        max(len(pct(report[k]['f1-score'])) for k in report if k not in ('accuracy', 'macro avg', 'weighted avg'))
    )
    sup_col = max(
        len('Support'),
        max(len(str(report[k]['support'])) for k in report if k not in ('accuracy', 'macro avg', 'weighted avg')),
        len(str(report['macro avg']['support']))
    )

    width = cls_col + 2 + prec_col + 2 + rec_col + 2 + f1_col + 2 + sup_col

    print("=" * width)
    print(f"{title}")
    print("=" * width)
    print()
    header = (
        f"{'Class':{cls_col}}  {'Precision':>{prec_col}}  {'Recall':>{rec_col}}  "
        f"{'F1-Score':>{f1_col}}  {'Support':>{sup_col}}"
    )
    print(header)
    print("-" * width)

    class_keys = [k for k in report.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
    class_keys = class_keys[:2]
    for k in class_keys:
        v = report[k]
        line = (
            f"{k:{cls_col}}  {pct(v['precision']):>{prec_col}}  {pct(v['recall']):>{rec_col}}  "
            f"{pct(v['f1-score']):>{f1_col}}  {str(v['support']):>{sup_col}}"
        )
        print(line)

    print("-" * width)
    acc_line = (
        f"{'Accuracy':{cls_col}}  {'':{prec_col}}  {'':{rec_col}}  "
        f"{pct(report['accuracy']):>{f1_col}}  {str(report['macro avg']['support']):>{sup_col}}"
    )
    print(acc_line)

    ma = report['macro avg']
    ma_line = (
        f"{'Macro Avg':{cls_col}}  {pct(ma['precision']):>{prec_col}}  {pct(ma['recall']):>{rec_col}}  "
        f"{pct(ma['f1-score']):>{f1_col}}  {str(ma['support']):>{sup_col}}"
    )
    print(ma_line)

    wa = report['weighted avg']
    wa_line = (
        f"{'Weighted Avg':{cls_col}}  {pct(wa['precision']):>{prec_col}}  {pct(wa['recall']):>{rec_col}}  "
        f"{pct(wa['f1-score']):>{f1_col}}  {str(wa['support']):>{sup_col}}"
    )
    print(wa_line)
    print("=" * width)
    print()

# ---------------------------------------------------------------------
# Main compute & pretty-print routine (keeps calculations unchanged)
# ---------------------------------------------------------------------
def compute_and_print(spam_path, vulgar_path, spam_threshold=SPAM_THRESHOLD, vulgar_threshold=VULGAR_THRESHOLD):
    print('\n=== ML STATISTICS REPORT ===\n')

    # -----------------------------
    # Spam dataset
    # -----------------------------
    try:
        spam_examples = load_labeled_examples(spam_path)
    except FileNotFoundError:
        print(f"Spam dataset not found: {spam_path}")
        spam_examples = []

    train_spam, test_spam = split_dataset_examples(spam_examples, 0.8, 42)
    spam_model = NaiveBayesSpamClassifier()
    train_model_from_examples(spam_model, train_spam)

    # evaluate
    train_stats = evaluate_spam_examples(spam_model, train_spam, threshold=spam_threshold)
    test_stats = evaluate_spam_examples(spam_model, test_spam, threshold=spam_threshold)

    # dataset summary (vocabulary size from trained model)
    vocab_size_spam = len(spam_model.vocabulary)
    print_dataset_summary("DATASET SUMMARY (SPAM)", spam_examples, vocab_size_spam, len(train_spam), len(test_spam), "spam", "ham")

    # Print train confusion + report
    print_confusion_matrix({'TP': train_stats['TP'], 'TN': train_stats['TN'], 'FP': train_stats['FP'], 'FN': train_stats['FN']},
                           pos_label="Spam", neg_label="Ham", header_title="CONFUSION MATRIX (TRAIN)")
    report_train = build_report_from_counts(train_stats['TP'], train_stats['TN'], train_stats['FP'], train_stats['FN'],
                                            pos_label="Spam", neg_label="Ham")
    print_classification_report(report_train, "CLASSIFICATION REPORT (TRAIN)")

    # Print test confusion + report
    print_confusion_matrix({'TP': test_stats['TP'], 'TN': test_stats['TN'], 'FP': test_stats['FP'], 'FN': test_stats['FN']},
                           pos_label="Spam", neg_label="Ham", header_title="CONFUSION MATRIX (TEST)")
    report_test = build_report_from_counts(test_stats['TP'], test_stats['TN'], test_stats['FP'], test_stats['FN'],
                                           pos_label="Spam", neg_label="Ham")
    print_classification_report(report_test, "CLASSIFICATION REPORT (TEST)")

    # -----------------------------
    # Vulgar dataset
    # -----------------------------
    try:
        vulgar_examples = load_labeled_examples(vulgar_path)
    except FileNotFoundError:
        print(f"Vulgar dataset not found: {vulgar_path}")
        vulgar_examples = []

    train_v, test_v = split_dataset_examples(vulgar_examples, 0.8, 42)
    vulgar_model = VulgarClassifier(alpha=VULGAR_ALPHA, vulgar_threshold=vulgar_threshold)
    train_model_from_examples(vulgar_model, train_v)

    # evaluate
    train_stats_v = evaluate_vulgar_examples(vulgar_model, train_v, threshold=vulgar_threshold)
    test_stats_v = evaluate_vulgar_examples(vulgar_model, test_v, threshold=vulgar_threshold)

    # dataset summary (vocabulary size from trained model)
    vocab_size_vulgar = len(vulgar_model.vocabulary)
    print_dataset_summary("VULGAR DATASET SUMMARY", vulgar_examples, vocab_size_vulgar, len(train_v), len(test_v), "vulgar", "clean")

    # Print train confusion + report
    print_confusion_matrix({'TP': train_stats_v['TP'], 'TN': train_stats_v['TN'], 'FP': train_stats_v['FP'], 'FN': train_stats_v['FN']},
                           pos_label="Vulgar", neg_label="Clean", header_title="CONFUSION MATRIX (TRAIN)")
    report_train_v = build_report_from_counts(train_stats_v['TP'], train_stats_v['TN'], train_stats_v['FP'], train_stats_v['FN'],
                                              pos_label="Vulgar", neg_label="Clean")
    print_classification_report(report_train_v, "CLASSIFICATION REPORT (TRAIN)")

    # Print test confusion + report
    print_confusion_matrix({'TP': test_stats_v['TP'], 'TN': test_stats_v['TN'], 'FP': test_stats_v['FP'], 'FN': test_stats_v['FN']},
                           pos_label="Vulgar", neg_label="Clean", header_title="CONFUSION MATRIX (TEST)")
    report_test_v = build_report_from_counts(test_stats_v['TP'], test_stats_v['TN'], test_stats_v['FP'], test_stats_v['FN'],
                                             pos_label="Vulgar", neg_label="Clean")
    print_classification_report(report_test_v, "CLASSIFICATION REPORT (TEST)")


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    spam_path = os.path.join(project_root, 'dataset', 'SpamCollection.txt')
    vulgar_path = os.path.join(project_root, 'dataset', 'vulgar_dataset.txt')
    compute_and_print(spam_path, vulgar_path)
