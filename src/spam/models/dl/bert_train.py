import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

class SpamHamDataset(Dataset):
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
    
    # Paths
    processed_dir = os.path.join(project_root, "data", "spam", "processed")
    models_dir = os.path.join(project_root, "models", "spam", "dl")
    os.makedirs(models_dir, exist_ok=True)
    
    data_path = os.path.join(processed_dir, "combined_data.csv")
    
    print("Loading raw text data for BERT...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text', 'spam'])
    X = df['text'].astype(str).values
    y = df['spam'].values
    
    print("Performing stratified 80-20 train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Initializing bert-base-uncased tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    print("Tokenizing training data...")
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)
    
    train_dataset = SpamHamDataset(train_encodings, y_train)
    test_dataset = SpamHamDataset(test_encodings, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    print("Initializing BERT model...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 3
    
    print(f"Training BERT model for {epochs} epochs on {device}...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss/len(train_loader):.4f}")
    
    # Save the model
    model_path = os.path.join(models_dir, "bert_model")
    print(f"Saving BERT model to {model_path}...")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Save test set for evaluation
    # For BERT, we'll save the raw test text and labels because saving encodings is huge
    test_features_path = os.path.join(processed_dir, "X_test_bert.npy")
    test_labels_path = os.path.join(processed_dir, "y_test_bert.npy")
    np.save(test_features_path, X_test)
    np.save(test_labels_path, y_test)
    
    print("BERT training complete!")

if __name__ == "__main__":
    main()