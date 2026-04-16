import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model():
    df = pd.read_csv("dataset.csv")

    X = df["text"]
    y = df["label"]

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    return model, vectorizer