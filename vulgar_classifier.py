"""
vulgar_classifier.py
====================
A second lightweight Naive Bayes classifier that operates on
character n-gram + word TF + special-char features extracted by
vulgar_features.py.

IMPORTANT: This classifier is COMPLETELY SEPARATE from the existing
NaiveBayesSpamClassifier in naive_bayes.py. It does NOT touch or
modify that class in any way.

MATHEMATICAL FOUNDATION
-----------------------
Multinomial Naive Bayes with Laplace smoothing.

For a message m with feature vector {f_1: v_1, f_2: v_2, ...}:

  log P(class | m) ∝ log P(class)
                   + Σ_i  v_i × log P(feature_i | class)

Where:
  P(class)           = (count(class) + 1) / (N + 2)         [prior]
  P(feature | class) = (sum_of_feature_vals_in_class + α)    [likelihood]
                     / (total_feature_weight_in_class + α×|V|)

α = Laplace smoothing constant (default 1.0).
|V| = vocabulary size (number of distinct features seen in training).

DECISION RULE (Part 4 of spec)
--------------------------------
  vulgar_probability = P(vulgar | message)
  IF vulgar_probability >= VULGAR_THRESHOLD  →  VULGAR
  ELSE                                       →  CLEAN

The probability is computed from log-scores using the softmax formula:
  P(vulgar) = exp(log_score_vulgar)
            / (exp(log_score_vulgar) + exp(log_score_clean))

To avoid overflow we use the log-sum-exp trick.

TRAINING DATA FORMAT
--------------------
Tab-separated file:  label \t message_text
Where label ∈ {vulgar, clean}

Example rows:
  vulgar    what the f**k is your problem
  clean     let me know when you are free to meet

"""

import math
import re
from collections import defaultdict

from vulgar_features import extract_features


class VulgarClassifier:
    """
    Multinomial Naive Bayes for vulgar / non-vulgar binary classification.

    Attributes
    ----------
    alpha : float
        Laplace smoothing constant.
    vulgar_threshold : float
        Minimum P(vulgar) required to classify as vulgar (0.0–1.0).
    vocabulary : set
        All feature keys seen during training.
    class_feature_sum : dict
        {class_label: {feature_key: total_weight}} accumulated from training.
    class_total_weight : dict
        {class_label: total weight of all features in that class}
    class_counts : dict
        {class_label: number of training examples in class}
    n_train : int
        Total number of training examples.
    trained : bool
        Flag set to True after train() completes.
    """

    # Class labels
    VULGAR = 'vulgar'
    CLEAN  = 'clean'
    LABELS = (VULGAR, CLEAN)

    def __init__(self, alpha: float = 1.0, vulgar_threshold: float = 0.60):
        """
        Parameters
        ----------
        alpha : float
            Laplace smoothing constant. Increase to smooth more aggressively.
        vulgar_threshold : float
            Probability threshold above which a message is flagged as vulgar.
            Default 0.60 per Part 4 of the spec.
        """
        self.alpha             = alpha
        self.vulgar_threshold  = vulgar_threshold

        # Training state
        self.vocabulary        : set              = set()
        self.class_feature_sum : dict             = {c: defaultdict(float) for c in self.LABELS}
        self.class_total_weight: dict[str, float] = {c: 0.0 for c in self.LABELS}
        self.class_counts      : dict[str, int]   = {c: 0   for c in self.LABELS}
        self.n_train           : int              = 0
        self.trained           : bool             = False

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, dataset_path: str) -> None:
        """
        Train from a tab-separated file: label\\tmessage

        Algorithm
        ---------
        1. For each training example, extract feature dict.
        2. Accumulate feature weights per class.
        3. Build global vocabulary (union of all features).
        4. (Log-likelihoods are computed lazily in predict to save memory.)

        Parameters
        ----------
        dataset_path : str
            Path to the vulgar training dataset.
        """
        self._reset()

        with open(dataset_path, 'r', encoding='utf-8') as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue

                # ── Parse label and text ──────────────────────────────────────
                parts = line.split('\t', 1)
                if len(parts) < 2:
                    continue
                label, text = parts[0].strip().lower(), parts[1].strip()

                # Normalise label: accept 'non_vulgar', 'nonvulgar', 'ham' as 'clean'
                if label in ('vulgar',):
                    label = self.VULGAR
                else:
                    label = self.CLEAN   # treat anything else as clean

                # ── Extract features ──────────────────────────────────────────
                feature_dict = extract_features(text)

                # ── Accumulate ────────────────────────────────────────────────
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
        """Clear all learned parameters."""
        self.vocabulary         = set()
        self.class_feature_sum  = {c: defaultdict(float) for c in self.LABELS}
        self.class_total_weight = {c: 0.0 for c in self.LABELS}
        self.class_counts       = {c: 0   for c in self.LABELS}
        self.n_train            = 0
        self.trained            = False

    # ──────────────────────────────────────────────────────────────────────────
    # Log-likelihood helper
    # ──────────────────────────────────────────────────────────────────────────

    def _log_likelihood(self, feature: str, label: str) -> float:
        """
        Compute log P(feature | label) with Laplace smoothing.

          P(feature | label) = (count(feature, label) + α)
                             / (total_weight(label) + α × |V|)
        """
        numerator   = self.class_feature_sum[label][feature] + self.alpha
        denominator = self.class_total_weight[label] + self.alpha * len(self.vocabulary)
        return math.log(numerator / denominator)

    # ──────────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def predict_proba(self, text: str) -> dict[str, float]:
        """
        Return {vulgar: P(vulgar|text), clean: P(clean|text)}.

        Uses log-sum-exp trick for numerical stability:
          P(vulgar) = softmax(log_score_vulgar, log_score_clean)[0]

        Parameters
        ----------
        text : str   Raw message text (normalization applied internally).

        Returns
        -------
        dict[str, float]
            Probabilities summing to 1.0.
        """
        if not self.trained:
            raise RuntimeError("VulgarClassifier has not been trained yet.")

        feature_dict = extract_features(text)

        # ── Compute log posteriors ─────────────────────────────────────────────
        log_scores: dict[str, float] = {}
        for label in self.LABELS:
            # log P(class) — Laplace-smoothed prior
            log_prior = math.log(
                (self.class_counts[label] + 1) / (self.n_train + len(self.LABELS))
            )
            log_score = log_prior
            for feat, val in feature_dict.items():
                log_score += val * self._log_likelihood(feat, label)
            log_scores[label] = log_score

        # ── Log-sum-exp normalisation ──────────────────────────────────────────
        # Prevents exp() overflow/underflow
        max_score = max(log_scores.values())
        exp_scores = {lbl: math.exp(s - max_score) for lbl, s in log_scores.items()}
        total      = sum(exp_scores.values())

        return {lbl: exp_scores[lbl] / total for lbl in self.LABELS}

    def predict(self, text: str) -> tuple[str, float]:
        """
        Classify a message.

        Returns
        -------
        (label, probability)
            label       : 'vulgar' or 'clean'
            probability : P(label | text)
        """
        proba = self.predict_proba(text)
        vulgar_prob = proba[self.VULGAR]

        # Part 4 decision rule: threshold-based
        if vulgar_prob >= self.vulgar_threshold:
            return self.VULGAR, vulgar_prob
        return self.CLEAN, proba[self.CLEAN]
