from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=8000
        )),
        ("model", LogisticRegression(max_iter=1000))
    ])
