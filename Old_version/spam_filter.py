"""
spam_filter.py  (ENHANCED)
==========================
Replaces the original spam_filter.py with a fully ML-based pipeline.

CHANGES vs ORIGINAL
-------------------
Original: Hybrid — Naive Bayes prediction (string label) + rule-based scoring.
Enhanced: Pure threshold-based ML decision using the *probabilities* the
          existing NaiveBayesSpamClassifier already computes internally.

PART 1 — Threshold-Based Spam Classification
---------------------------------------------
The existing naive_bayes.py predict() method returns a binary "spam"/"ham"
label. We extend its usage to extract the raw log-probabilities and convert
them to real probabilities, then apply a configurable threshold.

  spam_probability = P(spam | message)    (sigmoid of log-odds)
  confidence       = |P(spam) - P(ham)|   (distance from 0.50 boundary)

  IF spam_probability >= SPAM_THRESHOLD (0.70)  →  SPAM
  ELSE                                           →  HAM

WHY 0.70?
  The 0.50 midpoint just picks the more likely class. 0.70 requires the
  model to be at least 70 % confident before flagging — this reduces false
  positives on borderline messages while still catching clear spam.

PART 5 — Final Message Pipeline Priority
-----------------------------------------
  1. Run normaliser + spam classifier  → spam_label, spam_prob, confidence
  2. Run vulgar classifier             → vulgar_label, vulgar_prob
  3. Priority decision:
       VULGAR → block (regardless of spam status)
       SPAM   → store with spam flag
       CLEAN  → store normally

PART 6 — Actions
-----------------
  HAM    + CLEAN   →  folder = 'inbox',  spam_flag = 0, blocked = False
  SPAM   + CLEAN   →  folder = 'spam',   spam_flag = 1, blocked = False
  *      + VULGAR  →  folder = 'moderation', blocked = True,
                       returns warning "Inappropriate language detected."
"""

import math
from flask import current_app

# ── Thresholds (adjustable without retraining) ────────────────────────────────
# Raise SPAM_THRESHOLD → fewer spam flags (stricter).
# Lower  SPAM_THRESHOLD → more spam flags (looser).
SPAM_THRESHOLD   : float = 0.70   # Part 1
VULGAR_THRESHOLD : float = 0.70   # Part 4 (also set in VulgarClassifier)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: extract real probabilities from the existing NB model
# ─────────────────────────────────────────────────────────────────────────────

def _get_spam_probability(message: str) -> tuple[float, float]:
    """
    Compute P(spam) and P(ham) from the existing NaiveBayesSpamClassifier.

    The original model stores log-probability accumulators but does NOT
    expose them as probabilities. We replicate its log-score computation
    here to extract the probability WITHOUT touching the original class.

    Returns
    -------
    (spam_prob, ham_prob)
        Both sum to 1.0.
    """
    model = current_app.nb_model      # trained NaiveBayesSpamClassifier instance

    # ── Replicate the model's log-score accumulation ──────────────────────────
    # (mirrors the logic in naive_bayes.py predict() exactly)
    words = model.tokenize(message)
    total_msgs = model.spam_messages + model.ham_messages

    if total_msgs == 0:
        # Model untrained — return neutral probabilities
        return 0.5, 0.5

    log_spam = math.log(model.spam_messages / total_msgs)
    log_ham  = math.log(model.ham_messages  / total_msgs)

    for word in words:
        if word in model.STOP_WORDS:
            continue
        log_spam += math.log(model.word_prob(word, 'spam'))
        log_ham  += math.log(model.word_prob(word, 'ham'))

    # ── Convert log-scores to probabilities via log-sum-exp ───────────────────
    # P(spam) = exp(log_spam) / (exp(log_spam) + exp(log_ham))
    # Numerical stability: subtract max before exp.
    max_log = max(log_spam, log_ham)
    exp_spam = math.exp(log_spam - max_log)
    exp_ham  = math.exp(log_ham  - max_log)
    total    = exp_spam + exp_ham

    spam_prob = exp_spam / total
    ham_prob  = exp_ham  / total

    return spam_prob, ham_prob


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify_message(subject: str, body: str) -> dict:
    """
    Full moderation pipeline for a single message.

    Part 5 pipeline:
      Step 1: Preprocess (done inside each classifier)
      Step 2: Run spam classifier  → probability + threshold decision
      Step 3: Run vulgar classifier → probability + threshold decision
      Step 4: Apply priority: VULGAR > SPAM > CLEAN

    Parameters
    ----------
    subject : str   Message subject line.
    body    : str   Message body.

    Returns
    -------
    dict with keys:
        spam_probability  : float    P(spam | message)
        ham_probability   : float    P(ham  | message)
        confidence        : float    |P(spam) - P(ham)|
        is_spam           : bool     Threshold-based spam decision
        vulgar_probability: float    P(vulgar | message)
        is_vulgar         : bool     Threshold-based vulgar decision
        blocked           : bool     True if message should not be delivered
        folder            : str      'inbox' | 'spam' | 'moderation'
        spam_flag         : int      1 if spam else 0
        warning           : str | None  Warning message if blocked
        decision_reason   : str      Human-readable explanation
    """
    combined = f"{subject or ''} {body or ''}".strip()

    # ── Step 2: Spam classification ───────────────────────────────────────────
    spam_prob, ham_prob = _get_spam_probability(combined)
    confidence = abs(spam_prob - ham_prob)

    # Part 1 threshold decision
    is_spam_flag = spam_prob >= SPAM_THRESHOLD

    # ── Step 3: Vulgar classification ─────────────────────────────────────────
    vulgar_label, vulgar_prob = current_app.vulgar_model.predict(combined)
    is_vulgar = (vulgar_label == 'vulgar')

    # ── Step 4 & Part 5: Priority decision ───────────────────────────────────
    # VULGAR takes highest priority (blocks delivery)
    if is_vulgar:
        return {
            'spam_probability':   round(spam_prob,   4),
            'ham_probability':    round(ham_prob,     4),
            'confidence':         round(confidence,   4),
            'is_spam':            is_spam_flag,
            'vulgar_probability': round(vulgar_prob,  4),
            'is_vulgar':          True,
            'blocked':            True,
            'folder':             'moderation',
            'spam_flag':          0,
            'warning':            'Inappropriate language detected.',
            'decision_reason':    f'VULGAR (P={vulgar_prob:.3f} ≥ {VULGAR_THRESHOLD})'
        }

    # SPAM (not vulgar)
    if is_spam_flag:
        return {
            'spam_probability':   round(spam_prob,   4),
            'ham_probability':    round(ham_prob,     4),
            'confidence':         round(confidence,   4),
            'is_spam':            True,
            'vulgar_probability': round(vulgar_prob,  4),
            'is_vulgar':          False,
            'blocked':            False,
            'folder':             'spam',
            'spam_flag':          1,
            'warning':            None,
            'decision_reason':    f'SPAM (P={spam_prob:.3f} ≥ {SPAM_THRESHOLD})'
        }

    # CLEAN (ham + not vulgar)
    return {
        'spam_probability':   round(spam_prob,   4),
        'ham_probability':    round(ham_prob,     4),
        'confidence':         round(confidence,   4),
        'is_spam':            False,
        'vulgar_probability': round(vulgar_prob,  4),
        'is_vulgar':          False,
        'blocked':            False,
        'folder':             'inbox',
        'spam_flag':          0,
        'warning':            None,
        'decision_reason':    f'CLEAN (spam P={spam_prob:.3f}, vulgar P={vulgar_prob:.3f})'
    }


# ── Backward-compatible wrapper for old is_spam() call signature ──────────────
def is_spam(subject: str, body: str) -> tuple[bool, str]:
    """
    Backward-compatible shim for existing call sites in routes/mail.py.

    The original code calls: spam_detected, spam_reason = is_spam(subject, body)

    This wrapper preserves that interface while using the new pipeline
    internally. Existing code requires NO changes.
    """
    result = classify_message(subject, body)

    if result['blocked']:
        # Vulgar: treated as "spam" from the folder-routing perspective
        # so the caller stores it into a non-inbox folder
        return True, result['warning']

    return result['is_spam'], result['decision_reason']
