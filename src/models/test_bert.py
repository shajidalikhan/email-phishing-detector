import os
import torch
import warnings
from transformers import BertTokenizer, BertForSequenceClassification

# Suppress HuggingFace/PyTorch warnings for cleaner terminal output
warnings.filterwarnings("ignore")

def test_email(text):
    project_root = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(project_root, 'models', 'bert')
    
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please ensure you have downloaded and extracted the BERT model files.")
        return

    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    prediction_idx = torch.argmax(outputs.logits, dim=-1).item()
    confidence = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][prediction_idx].item() * 100
    
    result = "🚨 PHISHING EMAIL" if prediction_idx == 1 else "✅ SAFE EMAIL"
    
    print("\n" + "=" * 50)
    print("ANALYSIS RESULT:")
    print("=" * 50)
    print(f"Text: '{text}'")
    print("-" * 50)
    print(f"Result: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    print("\n" + "*" * 50)
    print("* Welcome to the Interactive BERT Phishing Tester *")
    print("*" * 50)
    
    while True:
        email_text = input("Paste an email to check (or type 'quit' to exit):\n> ")
        if email_text.lower() in ['quit', 'exit', 'q']:
            print("Exiting tester. Have a great day!")
            break
        if email_text.strip() == "":
            continue
            
        test_email(email_text)
