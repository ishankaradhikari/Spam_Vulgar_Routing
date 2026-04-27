import math
import re
from collections import defaultdict

class NaiveBayesSpamClassifier:

    STOP_WORDS = {
        "a","an","the","to","for","or","and","in","on","at",
        "is","it","you","your","of","this","that","with","from"
    }

    def __init__(self):
        self.spam_word_count = defaultdict(int)
        self.ham_word_count = defaultdict(int)

        self.total_spam_words = 0
        self.total_ham_words = 0

        self.spam_messages = 0
        self.ham_messages = 0

        self.vocabulary = set()

    # ─────────────────────────────────────────────
    # Tokenization
    # ─────────────────────────────────────────────
    def tokenize(self, message):
        message = message.lower()
        message = re.sub(r'[^a-zA-Z0-9\s]', '', message)
        return message.split()

    # ─────────────────────────────────────────────
    # Training
    # Dataset format:
    # spam    Win money now
    # ham     Hello how are you
    # ─────────────────────────────────────────────
    def train(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                parts = line.strip().split("\t", 1)
                if len(parts) < 2:
                    continue

                label, message = parts
                words = self.tokenize(message)

                if label.lower() == "spam":
                    self.spam_messages += 1
                    for word in words:
                        if word in self.STOP_WORDS:
                            continue
                        self.spam_word_count[word] += 1
                        self.total_spam_words += 1
                        self.vocabulary.add(word)

                else:
                    self.ham_messages += 1
                    for word in words:
                        if word in self.STOP_WORDS:
                            continue
                        self.ham_word_count[word] += 1
                        self.total_ham_words += 1
                        self.vocabulary.add(word)

        print("Naive Bayes Training Completed")
        print("Spam:", self.spam_messages, "Ham:", self.ham_messages)
        print("Vocabulary:", len(self.vocabulary))

    # ─────────────────────────────────────────────
    # Word Probability (Laplace smoothing)
    # ─────────────────────────────────────────────
    def word_prob(self, word, label):
        if label == "spam":
            word_count = self.spam_word_count[word]
            total_words = self.total_spam_words
        else:
            word_count = self.ham_word_count[word]
            total_words = self.total_ham_words

        return (word_count + 1) / (total_words + len(self.vocabulary))

    # ─────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────
    def predict(self, message):

        words = self.tokenize(message)

        total_msgs = self.spam_messages + self.ham_messages

        log_prob_spam = math.log(self.spam_messages / total_msgs)
        log_prob_ham = math.log(self.ham_messages / total_msgs)

        for word in words:
            if word in self.STOP_WORDS:
                continue
            log_prob_spam += math.log(self.word_prob(word, "spam"))
            log_prob_ham += math.log(self.word_prob(word, "ham"))

        return "spam" if log_prob_spam > log_prob_ham else "ham"