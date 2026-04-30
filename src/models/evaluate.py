from sklearn.metrics import classification_report
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def evaluate():
    df = pd.read_csv("../data/Phishing_Email.csv", engine='python', on_bad_lines='skip')
    df = df.dropna(subset=["Email Text"])

    df["label"] = df["Email Type"].map({
        "Safe Email": 0,
        "Phishing Email": 1
    })

    df = df.dropna(subset=["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["Email Text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    model = joblib.load("../models/phishing_model.pkl")
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

if __name__ == "__main__":
    evaluate()
