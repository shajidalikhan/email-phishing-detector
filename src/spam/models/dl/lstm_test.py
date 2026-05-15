import os
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

_TOKENIZER = None
_MODEL = None
MAX_LEN = 200

def _load_resources():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        tokenizer_path = os.path.join(project_root, "models", "spam", "dl", "lstm_tokenizer.pkl")
        model_path = os.path.join(project_root, "models", "spam", "dl", "lstm_model.keras")
        
        if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Model or Tokenizer not found. Please train the model first.")
            
        _TOKENIZER = joblib.load(tokenizer_path)
        _MODEL = tf.keras.models.load_model(model_path)

def predict_new_message(message: str):
    _load_resources()
    seq = _TOKENIZER.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    prob = float(_MODEL.predict(padded, verbose=0)[0][0])
    prediction = 1 if prob >= 0.5 else 0
    
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