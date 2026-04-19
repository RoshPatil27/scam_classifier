import pandas as pd

def train_model():
    df = pd.read_csv("dataset.csv")

    spam_words = set()
    ham_words = set()

    for _, row in df.iterrows():
        words = row["text"].lower().split()

        if row["label"] == 1:
            spam_words.update(words)
        else:
            ham_words.update(words)

    return spam_words, ham_words


def predict(message, spam_words, ham_words):
    words = message.lower().split()

    spam_score = sum(word in spam_words for word in words)
    ham_score = sum(word in ham_words for word in words)

    return 1 if spam_score > ham_score else 0