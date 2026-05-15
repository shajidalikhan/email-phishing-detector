import os
import sys
import numpy as np
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

sys.path.append(os.path.join(project_root, "src", "spam", "models"))
from evaluate_utils import save_evaluation_report

class SpamHamTestDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    processed_dir = os.path.join(project_root, "data", "spam", "processed")
    models_dir = os.path.join(project_root, "models", "spam", "dl", "bert_model")
    docs_dir = os.path.join(project_root, "docs", "spam", "model", "dl")
    
    test_features_path = os.path.join(processed_dir, "X_test_bert.npy")
    test_labels_path = os.path.join(processed_dir, "y_test_bert.npy")
    
    print("Loading test data...")
    X_test = np.load(test_features_path, allow_pickle=True)
    y_test = np.load(test_labels_path)
    
    print("Loading BERT model and tokenizer...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizerFast.from_pretrained(models_dir)
    model = BertForSequenceClassification.from_pretrained(models_dir)
    model.to(device)
    model.eval()
    
    print("Tokenizing test data...")
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)
    test_dataset = SpamHamTestDataset(test_encodings, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print("Generating predictions...")
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            
    y_pred = np.array(y_pred)
    
    hyperparams = {
        "Model Type": "bert-base-uncased",
        "Max Length": 512,
        "Optimizer": "AdamW",
        "Learning Rate": "5e-5",
        "Epochs": 3
    }
    
    test_size = len(y_test)
    train_size = int(test_size / 0.2 * 0.8)
    
    split_details = {
        "Split Strategy": "80-20 Stratified Split (random_state=42)",
        "Training Samples": train_size,
        "Testing Samples": test_size,
        "Total Dataset Size": train_size + test_size
    }
    
    print("Evaluating and saving report...")
    save_evaluation_report(
        model_name="BERT (bert-base-uncased)",
        y_true=y_test,
        y_pred=y_pred,
        hyperparams=hyperparams,
        split_details=split_details,
        output_dir=docs_dir,
        file_name="bert_report.md"
    )

if __name__ == "__main__":
    main()