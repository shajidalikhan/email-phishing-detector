import os
import joblib
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

_VECTORIZER = None
_MODEL = None

def _load_resources():
    global _VECTORIZER, _MODEL
    if _VECTORIZER is None or _MODEL is None:
        vectorizer_path = os.path.join(project_root, "models", "spam", "tfidf_vectorizer.pkl")
        model_path = os.path.join(project_root, "models", "spam", "ml", "svm.pkl")
        
        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Model or Vectorizer not found. Please train the model first.")
            
        _VECTORIZER = joblib.load(vectorizer_path)
        _MODEL = joblib.load(model_path)

def predict_new_message(message: str):
    _load_resources()
    features = _VECTORIZER.transform([message])
    prediction = _MODEL.predict(features)[0]
    
    # LinearSVC does not have predict_proba, use decision_function with sigmoid
    decision = _MODEL.decision_function(features)[0]
    prob = 1 / (1 + np.exp(-decision))
    
    confidence = prob * 100 if prediction == 1 else (1 - prob) * 100
    label = "Spam" if prediction == 1 else "Ham"
    
    return {
        "label": label,
        "confidence": confidence,
        "is_spam": bool(prediction == 1)
    }

if __name__ == "__main__":
    test_msg = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize."
    print(f"Testing message: '{test_msg}'")
    result = predict_new_message(test_msg)
    print(f"Result: {result['label']} (Confidence: {result['confidence']:.2f}%)")