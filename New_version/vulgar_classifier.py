"""
vulgar_classifier.py
====================
Multinomial Naive Bayes for vulgar / non-vulgar binary classification.

Uses character n-grams + word tokens extracted by vulgar_features.py.
Completely independent from the spam NaiveBayesSpamClassifier.
"""

import math
from collections import defaultdict
from vulgar_features import extract_features


class VulgarClassifier:

    VULGAR = 'vulgar'
    CLEAN  = 'clean'
    LABELS = (VULGAR, CLEAN)

    def __init__(self, alpha: float = 2.0, vulgar_threshold: float = 0.80):
        """
        alpha             : Laplace smoothing. Higher = more conservative (fewer false positives).
        vulgar_threshold  : Minimum P(vulgar) to flag. 0.80 means the model must be
                            80% confident before blocking — reduces false positives.
        """
        self.alpha            = alpha
        self.vulgar_threshold = vulgar_threshold

        self.vocabulary         : set              = set()
        self.class_feature_sum  : dict             = {c: defaultdict(float) for c in self.LABELS}
        self.class_total_weight : dict             = {c: 0.0 for c in self.LABELS}
        self.class_counts       : dict             = {c: 0   for c in self.LABELS}
        self.n_train            : int              = 0
        self.trained            : bool             = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, dataset_path: str) -> None:
        """
        Train from a tab-OR-space-separated file: label<sep>message

        Accepts both tab-delimited and whitespace-delimited lines so the
        full 602-line dataset is used (not just the 300 tab-delimited ones).
        """
        self._reset()

        with open(dataset_path, 'r', encoding='utf-8') as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue

                # Support both tab-separated and space-separated label lines
                if '\t' in line:
                    parts = line.split('\t', 1)
                else:
                    parts = line.split(None, 1)   # split on any whitespace, max 1 split

                if len(parts) < 2:
                    continue

                label = parts[0].strip().lower()
                text  = parts[1].strip()

                # Normalise label variants
                if label == 'vulgar':
                    label = self.VULGAR
                else:
                    label = self.CLEAN

                feature_dict = extract_features(text)

                self.class_counts[label] += 1
                self.n_train             += 1
                for feat, val in feature_dict.items():
                    self.class_feature_sum[label][feat] += val
                    self.class_total_weight[label]       += val
                    self.vocabulary.add(feat)

        self.trained = True
        print(f"[VulgarClassifier] Training complete.")
        print(f"  Vulgar examples : {self.class_counts[self.VULGAR]}")
        print(f"  Clean examples  : {self.class_counts[self.CLEAN]}")
        print(f"  Vocabulary size : {len(self.vocabulary)}")

    def _reset(self) -> None:
        self.vocabulary         = set()
        self.class_feature_sum  = {c: defaultdict(float) for c in self.LABELS}
        self.class_total_weight = {c: 0.0 for c in self.LABELS}
        self.class_counts       = {c: 0   for c in self.LABELS}
        self.n_train            = 0
        self.trained            = False

    # ── Log-likelihood ────────────────────────────────────────────────────────

    def _log_likelihood(self, feature: str, label: str) -> float:
        numerator   = self.class_feature_sum[label][feature] + self.alpha
        denominator = self.class_total_weight[label] + self.alpha * len(self.vocabulary)
        return math.log(numerator / denominator)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba(self, text: str) -> dict:
        if not self.trained:
            raise RuntimeError("VulgarClassifier has not been trained yet.")

        feature_dict = extract_features(text)

        log_scores = {}
        for label in self.LABELS:
            log_prior = math.log(
                (self.class_counts[label] + 1) / (self.n_train + len(self.LABELS))
            )
            log_score = log_prior
            for feat, val in feature_dict.items():
                log_score += val * self._log_likelihood(feat, label)
            log_scores[label] = log_score

        max_score  = max(log_scores.values())
        exp_scores = {lbl: math.exp(s - max_score) for lbl, s in log_scores.items()}
        total      = sum(exp_scores.values())

        return {lbl: exp_scores[lbl] / total for lbl in self.LABELS}

    def predict(self, text: str) -> tuple:
        proba       = self.predict_proba(text)
        vulgar_prob = proba[self.VULGAR]

        if vulgar_prob >= self.vulgar_threshold:
            return self.VULGAR, vulgar_prob
        return self.CLEAN, proba[self.CLEAN]
