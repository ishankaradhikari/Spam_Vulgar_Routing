"""Centralised ML configuration values.

This module defines the shared thresholds and smoothing constants used by the
spam and vulgar classifiers.  Keeping these values in one place ensures the
production app and CLI evaluation use the same decision boundaries.
"""

SPAM_THRESHOLD = 0.70
SUBJECT_SPAM_THRESHOLD = 0.50
VULGAR_THRESHOLD = 0.70
VULGAR_ALPHA = 2.0
