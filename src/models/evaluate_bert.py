import os
import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def predict(text, model_path):
    print(f"Loading model from {model_path}...")
    
    try:
        # Load the saved model and tokenizer from the local directory
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have downloaded the trained model files to the models/bert directory.")
        return

    # Put the model in evaluation mode
    model.eval()

    # Tokenize the input text
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get the predicted class index (0 for Safe, 1 for Phishing)
    prediction_idx = torch.argmax(outputs.logits, dim=-1).item()
    
    label_map = {0: "Safe Email", 1: "Phishing Email"}
    result = label_map[prediction_idx]
    
    print("\n--- Prediction Results ---")
    print(f"Input Text: {text}")
    print(f"Prediction: {result}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an email is phishing using the trained BERT model.")
    parser.add_argument("--text", type=str, required=True, help="The email text to classify.")
    args = parser.parse_args()
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_save_path = os.path.join(project_root, 'models', 'bert')
    
    predict(args.text, model_save_path)
