import os
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

_TOKENIZER = None
_MODEL = None
_DEVICE = None

def _load_resources():
    global _TOKENIZER, _MODEL, _DEVICE
    if _TOKENIZER is None or _MODEL is None:
        model_path = os.path.join(project_root, "models", "spam", "dl", "bert_model")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("BERT model directory not found. Please train the model first.")
            
        _DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        _TOKENIZER = BertTokenizerFast.from_pretrained(model_path)
        _MODEL = BertForSequenceClassification.from_pretrained(model_path)
        _MODEL.to(_DEVICE)
        _MODEL.eval()

def predict_new_message(message: str):
    _load_resources()
    
    inputs = _TOKENIZER(message, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = _MODEL(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    prediction = np.argmax(probs)
    confidence = probs[prediction] * 100
    
    label = "Spam" if prediction == 1 else "Ham"
    
    return {
        "label": label,
        "confidence": confidence,
        "is_spam": bool(prediction == 1)
    }

if __name__ == "__main__":
    import numpy as np
    test_msg = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize."
    print(f"Testing message: '{test_msg}'")
    result = predict_new_message(test_msg)
    print(f"Result: {result['label']} (Confidence: {result['confidence']:.2f}%)")