import joblib

def load_model():
    return joblib.load("../models/phishing_model.pkl")

def predict_email(text):
    model = load_model()
    pred = model.predict([text])[0]
    return "Phishing" if pred == 1 else "Safe"

if __name__ == "__main__":
    text = input("Enter email: ")
    print(predict_email(text))
