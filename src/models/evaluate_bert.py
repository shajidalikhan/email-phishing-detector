import os
import argparse
import torch
import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification

def get_model_details(model_path):
    try:
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)
        return config
    except:
        return {}

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=-1).item()

def predict_single(text, model_path):
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    prediction_idx = predict(text, model, tokenizer)
    label_map = {0: "Safe Email", 1: "Phishing Email"}
    result = label_map[prediction_idx]
    
    print("\n--- Prediction Results ---")
    print(f"Input Text: {text}")
    print(f"Prediction: {result}")
    return result

def evaluate_model(data_path, model_path, eval_samples, report_path):
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {data_path}")
        return

    df = df.dropna(subset=['Email Text', 'Email Type'])
    df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    if eval_samples > 0 and len(df) > eval_samples:
        print(f"Sampling {eval_samples} rows for evaluation...")
        df = df.sample(n=eval_samples, random_state=42)

    print(f"Loading model from {model_path}...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Evaluating {len(df)} samples. This may take a moment...")
    predictions = []
    texts = df['Email Text'].tolist()
    
    # Simple batch prediction loop
    for i, text in enumerate(texts):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{len(texts)}...")
        pred = predict(text, model, tokenizer)
        predictions.append(pred)

    true_labels = df['label'].tolist()
    
    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)
    cm = confusion_matrix(true_labels, predictions)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback if there's only 1 class in the sample
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(set(true_labels)) == 1:
            print("Warning: Sample contained only one class. Confusion matrix cannot be fully rendered.")

    # Get model parameters
    config = get_model_details(model_path)
    vocab_size = config.get("vocab_size", "Unknown")
    num_labels = config.get("num_labels", 2)
    max_position_embeddings = config.get("max_position_embeddings", 512)
    architectures = config.get("architectures", ["Unknown"])[0]

    report = f"""# BERT Model Evaluation Report
*Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## 1. Model Details & Parameters
* **Architecture:** `{architectures}`
* **Vocabulary Size:** `{vocab_size}`
* **Max Sequence Length Capacity:** `{max_position_embeddings}`
* **Number of Labels:** `{num_labels}`
* **Model Path:** `{model_path}`

## 2. Evaluation Dataset
* **Source:** `{data_path}`
* **Samples Evaluated:** `{len(df)}`

## 3. Performance Metrics
| Metric | Score |
| :--- | :--- |
| **Accuracy** | {acc:.4f} |
| **Precision** | {precision:.4f} |
| **Recall** | {recall:.4f} |
| **F1-Score** | {f1:.4f} |

## 4. Confusion Matrix

| | Predicted Safe (0) | Predicted Phishing (1) |
| :--- | :--- | :--- |
| **Actual Safe (0)** | {tn} (True Negatives) | {fp} (False Positives) |
| **Actual Phishing (1)** | {fn} (False Negatives) | {tp} (True Positives) |

"""
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"\nEvaluation Complete! Report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or Predict using the trained BERT model.")
    parser.add_argument("--text", type=str, default=None, help="The email text to classify.")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on a dataset.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset for evaluation.")
    parser.add_argument("--eval_samples", type=int, default=1000, help="Number of samples to evaluate (0 for all).")
    args = parser.parse_args()
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_save_path = os.path.join(project_root, 'models', 'bert')
    
    if args.evaluate:
        data_path = args.data_path if args.data_path else os.path.join(project_root, 'data', 'external', 'Phishing_Email.csv')
        report_path = os.path.join(project_root, 'docs', 'bert_evaluation_report.md')
        evaluate_model(data_path, model_save_path, args.eval_samples, report_path)
    elif args.text:
        predict_single(args.text, model_save_path)
    else:
        print("Please specify either --text \"your_email\" or --evaluate")
